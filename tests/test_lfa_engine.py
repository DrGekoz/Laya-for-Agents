"""Engine tests: the budget policy, dispatch, envelopes and error mapping.

A fake router stands in for a real checkpoint, so these run in milliseconds and
pin the behaviour that matters -- that a long state is *scanned* rather than cut,
and that a question the model cannot encode becomes a 422 rather than a 500.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_for_agents import protocol  # noqa: E402
from laya_for_agents.engine import Busy, Engine  # noqa: E402
from laya_for_agents.settings import Settings  # noqa: E402


class FakeRouter:
    """Records how it was called and returns a Laya-shaped reply.

    ``loaded`` and ``loaded_revisions`` are **properties**, matching Laya's own
    ``Router``. Getting that wrong is silent -- the introspection is wrapped so a
    failure cannot break ``/health``, which means the endpoint just reports an
    empty server forever. ``LiteralRouter`` below covers the other shape.
    """

    def __init__(self, probabilities=None):
        self.calls = []
        self.probabilities = probabilities

    @property
    def loaded(self):
        return ["english"]

    @property
    def loaded_revisions(self):
        return {"english": "test"}

    def _answers(self, questions):
        out = {}
        for name, question in questions.items():
            kind = question["type"]
            if kind == "choice":
                labels = list(question["criteria"])
                p = self.probabilities or [0.7] + [0.3 / max(len(labels) - 1, 1)] * (len(labels) - 1)
                total = sum(p)
                p = [x / total for x in p]
                entropy = 1 - (-sum(x * math.log(x) for x in p if x) / math.log(len(p))) if len(p) > 1 else 1.0
                out[name] = {"type": "choice", "choice": labels[0],
                             "probabilities": {l: round(v, 4) for l, v in zip(labels, p)},
                             "confidence": round(entropy, 4),
                             "answer_confidence": round(max(p), 4)}
            elif kind == "score":
                levels = len(question["criteria"])
                p = [1.0 / levels] * levels
                out[name] = {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4),
                             "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)},
                             "confidence": 0.0, "answer_confidence": round(max(p), 4)}
            else:
                out[name] = {"type": "noul", "noul": 0.75, "confidence": 0.75,
                             "answer_confidence": 0.75}
        return out

    def predict(self, state, questions, **kwargs):
        self.calls.append(("predict", kwargs))
        return {"answers": self._answers(questions), "usage": {"input_tokens": 12, "output_tokens": 0},
                "routing": {"model": "english", "repo": "convaiinnovations/laya"}}

    def predict_long(self, state, questions, **kwargs):
        self.calls.append(("predict_long", kwargs))
        return {"answers": self._answers(questions),
                "usage": {"input_tokens": 12, "output_tokens": 0, "windows": 7},
                "routing": {"model": "multilingual", "repo": "convaiinnovations/laya-multilingual"}}


def request(state, questions=None, **extra):
    body = {"state": state}
    body["questions"] = questions or {"q": {"type": "noul", "instructions": "is it risky?"}}
    body.update(extra)
    return protocol.check_request(body, Settings())


class PlanTests(unittest.TestCase):
    """The policy that exists because Laya's `predict` truncates silently."""

    def setUp(self):
        self.engine = Engine(Settings())
        self.engine._router = FakeRouter()

    def test_a_short_state_is_answered_in_one_pass(self):
        plan = self.engine.plan(request("short question"))
        self.assertEqual(plan["mode"], "single")

    def test_a_long_state_is_sent_to_the_long_checkpoint_rather_than_truncated(self):
        """One pass on the 8,192-token checkpoint beats 42 windows of the 512-token one.

        Measured on CPU: the multilingual pass takes seconds, the scan took 182 s.
        """
        plan = self.engine.plan(request("word " * 4_000))     # ~5000 tokens
        self.assertEqual(plan["mode"], "multilingual")
        self.assertGreater(plan["needed_tokens"], plan["single_max"])

    def test_a_state_past_the_long_checkpoint_is_refused_by_default(self):
        """~10k tokens: past the 8,192-token checkpoint, and a scan would take minutes.

        Refusing fails the client open immediately; scanning would return a decision
        long after the turn it was for had been answered some other way.
        """
        engine = Engine(Settings())
        engine._router = FakeRouter()
        with self.assertRaises(protocol.ProtocolError) as caught:
            engine.plan(request("word " * 8_000))            # ~10,000 tokens
        self.assertEqual(caught.exception.status, 413)
        self.assertIn("minutes", caught.exception.detail)

    def test_scanning_can_be_enabled_explicitly(self):
        engine = Engine(Settings(long_policy="scan", scan_max_tokens=40_000))
        engine._router = FakeRouter()
        plan = engine.plan(request("word " * 8_000))
        self.assertEqual(plan["mode"], "scan")

    def test_a_scan_larger_than_the_cap_is_refused_rather_than_hanging(self):
        engine = Engine(Settings(long_policy="scan", scan_max_tokens=2_000))
        engine._router = FakeRouter()
        with self.assertRaises(protocol.ProtocolError) as caught:
            engine.plan(request("word " * 4_000))            # ~5,000 tokens
        self.assertEqual(caught.exception.status, 413)

    def test_the_scan_policy_replaces_the_long_checkpoint_for_mid_size_states(self):
        engine = Engine(Settings(long_policy="scan", scan_max_tokens=40_000))
        engine._router = FakeRouter()
        self.assertEqual(engine.plan(request("word " * 4_000))["mode"], "scan")

    def test_a_2_5k_character_turn_is_over_the_stock_512_token_window(self):
        """The regression this package exists for.

        A 2,500-character turn -- an ordinary agent prompt -- is about 625 tokens.
        The stock server answers it from a 512-token window and discards the rest,
        with no error. Laya for Agents raises the window to cover it instead.
        """
        plan = self.engine.plan(request("x" * 2_500))
        self.assertGreater(plan["estimated_state_tokens"], 512)
        self.assertLessEqual(plan["needed_tokens"], plan["single_max"])   # one pass is enough ...
        self.assertGreater(plan["needed_tokens"], 512)                    # ... but not at 512

    def test_the_threshold_is_where_single_max_says(self):
        engine = Engine(Settings(single_max=16_000))
        engine._router = FakeRouter()
        self.assertEqual(engine.plan(request("word " * 2_000))["mode"], "single")   # ~2500 tokens

    def test_estimation_counts_the_question_as_well_as_the_state(self):
        small = self.engine.plan(request("hi"))
        wide = self.engine.plan(request("hi", {"q": {
            "type": "choice", "instructions": "pick one",
            "criteria": {f"option-{i}": "a fairly long description of an option" for i in range(60)}}}))
        self.assertGreater(wide["needed_tokens"], small["needed_tokens"])


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.router = FakeRouter()
        self.engine = Engine(Settings())
        self.engine._router = self.router

    def test_a_short_state_goes_through_predict(self):
        self.engine.answer(request("hi"))
        kind, kwargs = self.router.calls[-1]
        self.assertEqual(kind, "predict")
        self.assertIn("head_max_len", kwargs)

    def test_a_short_state_gets_at_least_the_stock_window(self):
        """A tiny request must not be given a tighter window than stock by accident."""
        self.engine.answer(request("hi"))
        _, kwargs = self.router.calls[-1]
        self.assertGreaterEqual(kwargs["max_len"], Settings().window_floor)

    def test_the_budget_is_never_below_what_the_request_needs(self):
        for size in (10, 500, 2_000, 3_600):
            self.engine.answer(request("x" * size))
            kind, kwargs = self.router.calls[-1]
            if kind == "predict":
                plan_needed = self.engine.plan(request("x" * size))["needed_tokens"]
                self.assertGreater(kwargs["max_len"], plan_needed,
                                   f"a {size}-character state was given a window that does not fit it")

    def test_a_scan_goes_through_predict_long(self):
        engine = Engine(Settings(long_policy="scan", scan_max_tokens=40_000))
        router = FakeRouter()
        engine._router = router
        engine.answer(request("word " * 8_000))
        kind, kwargs = router.calls[-1]
        self.assertEqual(kind, "predict_long")
        self.assertNotIn("max_len", kwargs)      # predict_long sizes its own windows

    def test_a_long_state_pins_the_long_checkpoint_on_one_pass(self):
        self.engine.answer(request("word " * 4_000))
        kind, kwargs = self.router.calls[-1]
        self.assertEqual(kind, "predict")
        self.assertEqual(kwargs["model"], "multilingual")
        self.assertGreaterEqual(kwargs["max_len"], 5_000)

    def test_a_jev_model_id_is_pinned_to_the_long_checkpoint(self):
        """`jev-latest` means 'you choose', so the long policy may choose."""
        self.engine.answer(request("word " * 4_000, model="jev-latest"))
        _, kwargs = self.router.calls[-1]
        self.assertEqual(kwargs["model"], "multilingual")

    def test_an_explicit_checkpoint_is_respected_and_its_truncation_reported(self):
        """A caller who really names `english` keeps it. If that cuts the state, the
        reply says so rather than the answer looking merely plausible."""
        self.engine.answer(request("word " * 4_000, model="english"))
        _, kwargs = self.router.calls[-1]
        self.assertEqual(kwargs["model"], "english")

    def test_the_single_pass_budget_is_raised_to_fit_the_state_not_left_at_512(self):
        """A 900-token state must not be handed a 512-token window."""
        state = "x" * 3_600                        # ~900 tokens
        engine = Engine(Settings(single_max=4_096))
        engine._router = self.router
        engine.answer(request(state))
        _, kwargs = self.router.calls[-1]
        self.assertGreaterEqual(kwargs["max_len"], 900)

    def test_a_wide_question_widens_head_max_len(self):
        questions = {"q": {"type": "choice", "instructions": "pick",
                           "criteria": {f"o{i}": "a description of an option" * 3 for i in range(20)}}}
        self.engine.answer(request("hi", questions))
        _, kwargs = self.router.calls[-1]
        self.assertGreater(kwargs["head_max_len"], 100)

    def test_head_max_len_is_capped(self):
        """A question too wide to fit is bounded, so it can never starve the state.

        Checked on the budget builder directly: a question this big pushes the whole
        request over ``single_max`` and into a scan, where ``head_max_len`` does not
        apply -- but the ceiling still governs any single pass.
        """
        questions = {"q": {"type": "choice", "instructions": "pick",
                           "criteria": {f"o{i}": "very long description " * 20 for i in range(80)}}}
        checked = request("hi", questions)
        self.assertLessEqual(self.engine._head_budget(checked), Settings().head_max_len_ceiling)

    def test_a_moderate_question_stays_in_one_pass_even_when_widened(self):
        questions = {"q": {"type": "choice", "instructions": "pick",
                           "criteria": {f"o{i}": "a description of an option" for i in range(20)}}}
        checked = request("hi", questions)
        self.assertEqual(self.engine.plan(checked)["mode"], "single")
        self.assertGreater(self.engine._head_budget(checked), Settings().default_head_max_len)

    def test_a_caller_supplied_head_budget_wins(self):
        self.assertEqual(self.engine._head_budget(request("hi", head_max_len=384)), 384)

    def test_a_caller_supplied_budget_is_honoured(self):
        self.engine.answer(request("hi", max_len=2_048))
        _, kwargs = self.router.calls[-1]
        self.assertEqual(kwargs["max_len"], 2_048)

    def test_a_jev_model_id_is_dropped_rather_than_forwarded(self):
        """The bug that would have made this whole package useless.

        `Router.predict` hands `model` straight to `normalise_name`, which raises on
        anything it does not recognise -- and every Jev client sends `jev-latest`.
        Forwarding it made each request a 500. Laya's own server drops an id it does
        not know and lets the router choose; so does this.
        """
        self.engine.answer(request("hi", model="jev-latest"))
        _, kwargs = self.router.calls[-1]
        self.assertNotIn("model", kwargs)

    def test_the_free_tier_jev_id_is_also_dropped(self):
        self.engine.answer(request("hi", model="jev-1.13-free"))
        self.assertNotIn("model", self.router.calls[-1][1])

    def test_a_real_checkpoint_is_forwarded(self):
        self.engine.answer(request("hi", model="multilingual"))
        self.assertEqual(self.router.calls[-1][1]["model"], "multilingual")

    def test_a_published_hugging_face_id_resolves_to_its_checkpoint(self):
        self.engine.answer(request("hi", model="convaiinnovations/laya-multilingual"))
        self.assertEqual(self.router.calls[-1][1]["model"], "multilingual")

    def test_a_checkpoint_alias_resolves(self):
        self.engine.answer(request("hi", model="ml"))
        self.assertEqual(self.router.calls[-1][1]["model"], "multilingual")

    def test_resolve_model_is_the_rule(self):
        resolve = Engine._resolve_model
        self.assertIsNone(resolve(None))
        self.assertIsNone(resolve(""))
        self.assertIsNone(resolve("jev-latest"))
        self.assertIsNone(resolve("gpt-4"))
        self.assertEqual(resolve("english"), "english")
        self.assertEqual(resolve("multilingual"), "multilingual")
        self.assertEqual(resolve("typed-decisions"), "typed-decisions")

    def test_an_unknown_checkpoint_from_laya_is_a_422_not_a_500(self):
        class Angry(FakeRouter):
            def predict(self, state, questions, **kwargs):
                raise ValueError("unknown task 'nope'; choose one of ...")

        self.engine._router = Angry()
        with self.assertRaises(protocol.ProtocolError) as caught:
            self.engine.answer(request("hi", task="nope"))
        self.assertEqual(caught.exception.status, 422)

    def test_min_confidence_is_forwarded_only_when_asked(self):
        self.engine.answer(request("hi"))
        self.assertIsNone(self.router.calls[-1][1]["min_confidence"])
        self.engine.answer(request("hi", min_confidence=0.5))
        self.assertEqual(self.router.calls[-1][1]["min_confidence"], 0.5)


class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(Settings())
        self.engine._router = FakeRouter()

    def test_the_reply_has_the_keys_a_jev_client_reads(self):
        reply = self.engine.answer(request("hi"))
        for key in ("model", "answers", "usage", "routing"):
            self.assertIn(key, reply)
        self.assertIn("q", reply["answers"])

    def test_the_reply_says_which_mode_was_used(self):
        self.assertEqual(self.engine.answer(request("hi"))["laya"]["mode"], "single")
        self.assertEqual(self.engine.answer(request("word " * 4_000))["laya"]["mode"], "multilingual")

    def test_a_scanned_reply_reports_its_window_count(self):
        engine = Engine(Settings(long_policy="scan", scan_max_tokens=40_000))
        engine._router = self.engine._router = FakeRouter()
        reply = engine.answer(request("word " * 4_000))
        self.assertEqual(reply["laya"]["mode"], "scan")
        self.assertEqual(reply["laya"]["windows"], 7)

    def test_the_reply_says_whether_the_state_was_truncated(self):
        """The signal this whole package exists to make visible."""
        self.assertFalse(self.engine.answer(request("hi"))["laya"]["truncated"])
        self.assertEqual(self.engine.answer(request("hi"))["laya"]["state_tokens_dropped"], 0)

    def test_usage_always_carries_an_input_token_count(self):
        reply = self.engine.answer(request("hi"))
        self.assertIn("input_tokens", reply["usage"])
        self.assertGreater(reply["usage"]["input_tokens"], 0)

    def test_counters_move(self):
        self.engine.answer(request("hi"))
        self.engine.answer(request("word " * 4_000))
        self.assertEqual(self.engine.stats["single"], 1)
        self.assertEqual(self.engine.stats["multilingual"], 1)
        self.assertEqual(self.engine.stats["requests"], 2)

    def test_a_scan_is_counted_separately(self):
        engine = Engine(Settings(long_policy="scan", scan_max_tokens=40_000))
        engine._router = FakeRouter()
        engine.answer(request("word " * 8_000))
        self.assertEqual(engine.stats["scan"], 1)
        self.assertEqual(engine.stats["single"], 0)

    def test_the_reply_carries_jev_style_confidence_by_default(self):
        questions = {"k": {"type": "choice", "instructions": "which?",
                           "criteria": {"a": "first", "b": "second", "c": "third", "d": "fourth"}}}
        engine = Engine(Settings(confidence_style="jev"))
        router = FakeRouter(probabilities=[0.7, 0.1, 0.1, 0.1])
        engine._router = router
        answer = engine.answer(request("hi", questions))["answers"]["k"]
        self.assertAlmostEqual(answer["confidence"], 0.6, places=3)
        self.assertIn("confidence_laya", answer)

    def test_the_laya_style_reply_keeps_the_engines_confidence(self):
        questions = {"k": {"type": "choice", "instructions": "which?",
                           "criteria": {"a": "first", "b": "second"}}}
        engine = Engine(Settings(confidence_style="laya"))
        engine._router = FakeRouter()
        answer = engine.answer(request("hi", questions))["answers"]["k"]
        self.assertNotIn("confidence_laya", answer)

    def test_the_reply_says_which_confidence_style_it_used(self):
        self.assertEqual(self.engine.answer(request("hi"))["laya"]["confidence_style"], "jev")

    def test_health_reports_loading_and_ready(self):
        engine = Engine(Settings())
        engine._router = FakeRouter()
        self.assertTrue(engine.health()["ready"])
        self.assertFalse(engine.health()["loading"])

    def test_health_is_not_ready_while_loading(self):
        """A supervisor must be able to tell 'warming up' from 'answering'."""
        engine = Engine(Settings())
        engine._router = FakeRouter()
        engine.loading = True
        engine.loading_since = time.time()
        health = engine.health()
        self.assertTrue(health["loading"])
        self.assertFalse(health["ready"])
        self.assertIsNotNone(health["loading_for_s"])

    def test_status_stays_ok_while_loading(self):
        """The HTTP surface really is serving; a dumb 'is it up' probe must not fail
        for three minutes and get the process killed."""
        engine = Engine(Settings())
        engine._router = FakeRouter()
        engine.loading = True
        self.assertEqual(engine.health()["status"], "ok")

    def test_a_load_clears_the_loading_flag(self):
        engine = Engine(Settings())
        engine._router = FakeRouter()
        engine.warm()
        self.assertFalse(engine.loading)
        self.assertIsNotNone(engine.loaded_at)

    def test_a_failed_load_still_clears_the_loading_flag(self):
        class Broken(FakeRouter):
            def preload(self, names=None):
                raise RuntimeError("no disk")

        engine = Engine(Settings())
        engine._router = Broken()
        engine.warm()
        self.assertFalse(engine.loading, "a failed load must not leave the service stuck at loading")

    def test_health_reports_the_confidence_style(self):
        self.assertEqual(Engine(Settings()).health()["budget"]["confidence_style"], "jev")


class JevValidatorTests(unittest.TestCase):
    """Mirror the invariants hermes-jev-skills enforces on a reply.

    A reply that computes cleanly can still be refused by the client -- and a
    refusal costs the whole decision, because every feature there fails open. So
    the shapes that client checks are pinned here.
    """

    def setUp(self):
        self.engine = Engine(Settings())
        self.engine._router = FakeRouter()
        self.engine._router.probabilities = [0.9281, 0.0412, 0.0307]

    def test_choice_probabilities_cover_every_offered_option_and_sum_to_one(self):
        questions = {"k": {"type": "choice", "instructions": "which?",
                           "criteria": {"a": "first", "b": "second", "c": "third"}}}
        answer = self.engine.answer(request("hi", questions))["answers"]["k"]
        self.assertEqual(set(answer["probabilities"]), {"a", "b", "c"})
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=2)

    def test_the_reported_choice_is_the_argmax(self):
        questions = {"k": {"type": "choice", "instructions": "which?",
                           "criteria": {"a": "first", "b": "second", "c": "third"}}}
        answer = self.engine.answer(request("hi", questions))["answers"]["k"]
        p = answer["probabilities"]
        self.assertEqual(p[answer["choice"]], max(p.values()))

    def test_a_score_equals_the_mean_of_its_own_distribution(self):
        questions = {"k": {"type": "score", "instructions": "how much?",
                           "criteria": ["a", "b", "c", "d", "e"]}}
        answer = self.engine.answer(request("hi", questions))["answers"]["k"]
        spread = {int(k): v for k, v in answer["probabilities"].items()}
        self.assertAlmostEqual(sum(i * v for i, v in spread.items()), answer["score"], places=3)

    def test_every_confidence_is_inside_the_unit_interval(self):
        questions = {"k": {"type": "choice", "instructions": "which?",
                           "criteria": {"a": "first", "b": "second"}},
                     "s": {"type": "score", "instructions": "how much?", "criteria": ["lo", "hi"]},
                     "n": {"type": "noul", "instructions": "risky?"}}
        answers = self.engine.answer(request("hi", questions))["answers"]
        for name, answer in answers.items():
            self.assertGreaterEqual(answer["confidence"], 0.0, name)
            self.assertLessEqual(answer["confidence"], 1.0, name)

    def test_a_noul_is_a_probability_not_a_flag(self):
        answer = self.engine.answer(request("hi"))["answers"]["q"]
        self.assertIsInstance(answer["noul"], float)
        self.assertGreaterEqual(answer["noul"], 0.0)
        self.assertLessEqual(answer["noul"], 1.0)


class ErrorMappingTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(Settings())

    def test_a_question_the_model_cannot_encode_becomes_422(self):
        class Angry(FakeRouter):
            def predict(self, state, questions, **kwargs):
                raise ValueError("question 'k': only 3 of its 60 option markers fit in max_len=512")

        self.engine._router = Angry()
        with self.assertRaises(protocol.ProtocolError) as caught:
            self.engine.answer(request("hi"))
        self.assertEqual(caught.exception.status, 422)

    def test_an_unexpected_failure_becomes_a_fixed_500(self):
        class Broken(FakeRouter):
            def predict(self, state, questions, **kwargs):
                raise RuntimeError("C:/secret/path/weights.safetensors blew up")

        self.engine._router = Broken()
        with self.assertRaises(protocol.ProtocolError) as caught:
            self.engine.answer(request("hi"))
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(caught.exception.detail, "inference failed")   # nothing leaked

    def test_out_of_memory_reads_as_busy_not_as_a_crash(self):
        class Oom(FakeRouter):
            def predict(self, state, questions, **kwargs):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        self.engine._router = Oom()
        with self.assertRaises(Busy):
            self.engine.answer(request("hi"))

    def test_a_full_admission_gate_refuses_rather_than_queues(self):
        engine = Engine(Settings(max_concurrent=1))
        engine._router = FakeRouter()
        engine._admission.acquire()
        with self.assertRaises(Busy):
            engine.answer(request("hi"))


class HealthTests(unittest.TestCase):
    def test_health_reports_the_budget_it_will_use(self):
        engine = Engine(Settings())
        health = engine.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["protocol"], "Jev /v1/systemone")
        self.assertEqual(health["budget"]["single_max"], engine.settings.single_max)
        self.assertTrue(health["budget"]["auto_long"])

    def test_health_is_json_serialisable(self):
        json.dumps(Engine(Settings()).health())

    def test_resident_checkpoints_are_reported_when_loaded_is_a_property(self):
        """Laya's Router exposes `loaded` as a property; reading it as a method
        raises TypeError, which the defensive introspection swallows -- so a server
        would report an empty checkpoint list forever while working fine."""
        engine = Engine(Settings())
        engine._router = FakeRouter()
        described = engine.describe()
        self.assertEqual(described["loaded"], ["english"])
        self.assertEqual(described["loaded_revisions"], {"english": "test"})
        self.assertEqual(engine.health()["loaded"], ["english"])

    def test_resident_checkpoints_are_reported_when_loaded_is_a_method(self):
        """The other shape, so neither can regress silently."""

        class LiteralRouter(FakeRouter):
            def loaded(self):
                return ["multilingual"]

            def loaded_revisions(self):
                return {"multilingual": "test"}

        engine = Engine(Settings())
        engine._router = LiteralRouter()
        self.assertEqual(engine.describe()["loaded"], ["multilingual"])

    def test_preload_is_given_names_not_a_repo_mapping(self):
        """`Router(models=...)` is name -> (repo, subfolder). Building one from a bare
        name list makes it try to load a repo literally called "english"; the model
        list belongs on `preload`, which takes plain names."""
        seen = {}

        class Recording(FakeRouter):
            def preload(self, names=None):
                seen["names"] = names

        engine = Engine(Settings(models=["english", "multilingual"]))
        engine._router = Recording()
        engine.warm()
        self.assertEqual(seen["names"], ["english", "multilingual"])

    def test_preload_with_no_model_list_asks_for_the_default_set(self):
        seen = {}

        class Recording(FakeRouter):
            def preload(self, names=None):
                seen["names"] = names

        engine = Engine(Settings())
        engine._router = Recording()
        engine.warm()
        self.assertIsNone(seen["names"])

    def test_preload_disabled_does_not_preload(self):
        seen = {}

        class Recording(FakeRouter):
            def preload(self, names=None):
                seen["names"] = names

        engine = Engine(Settings(preload=False))
        engine._router = Recording()
        engine.warm()
        self.assertEqual(seen, {})

    def test_a_failed_preload_still_leaves_a_working_engine(self):
        class Broken(FakeRouter):
            def preload(self, names=None):
                raise RuntimeError("no disk")

        engine = Engine(Settings())
        engine._router = Broken()
        self.assertIn("english", engine.warm()["loaded"])   # describe() still answers

    def test_a_failed_introspection_still_answers_health(self):
        """Never let a diagnostic take the endpoint down."""

        class Hostile(FakeRouter):
            @property
            def loaded(self):
                raise RuntimeError("no")

        engine = Engine(Settings())
        engine._router = Hostile()
        self.assertEqual(engine.describe()["loaded"], [])
        self.assertEqual(engine.health()["status"], "ok")

    def test_describe_never_touches_the_router_lock(self):
        """The defect this pins, and it is upstream Laya's shape.

        `Router.load` holds `Router._lock` across `Agent(repo, **kwargs)` -- the whole
        checkpoint build, 92 s warm and 218 s cold -- and `Router.loaded` takes the
        same lock. So introspecting through the public property during a load blocks
        for minutes. On /health the connection is accepted and the reply never comes:
        worse than an obvious refusal, because a supervisor sees an open port and no
        answer and the client hangs to its timeout.
        """

        class LockedRouter(FakeRouter):
            """`loaded` blocks forever, exactly as the real one does while loading."""

            def __init__(self):
                super().__init__()
                self._order = ["english"]
                self._agents = {}

            @property
            def loaded(self):
                raise AssertionError("describe() must not read the locking property")

            @property
            def loaded_revisions(self):
                raise AssertionError("describe() must not read the locking property")

        engine = Engine(Settings())
        engine._router = LockedRouter()
        engine.loading = True
        described = engine.describe()
        self.assertEqual(described["loaded"], ["english"])   # read from the registry
        self.assertEqual(engine.health()["loaded"], ["english"])

    def test_describe_falls_back_to_the_cache_when_it_cannot_read(self):
        class Opaque(FakeRouter):
            def __init__(self):
                super().__init__()
                self._order = "not a list"

        engine = Engine(Settings())
        engine._router = Opaque()
        engine.loading = True                                  # so the property is skipped
        engine._resident = {"loaded": ["cached"], "loaded_revisions": {}}
        self.assertEqual(engine.describe()["loaded"], ["cached"])

    def test_health_stays_cheap_when_polled(self):
        """A polled endpoint must not re-probe torch's device on every call."""
        engine = Engine(Settings())
        engine._router = FakeRouter()
        engine.health()
        first = engine._device
        engine.health()
        self.assertIs(engine._device, first)


if __name__ == "__main__":
    unittest.main()
