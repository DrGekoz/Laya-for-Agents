"""Wire-protocol tests: validation, token estimation, envelopes.

Pure stdlib, no torch, no server. These are the checks that decide whether a
malformed request becomes a precise HTTP status (which a Jev client fails open
on) or a silently odd decision.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_for_agents import protocol  # noqa: E402
from laya_for_agents.settings import Settings  # noqa: E402


class EstimationTests(unittest.TestCase):
    def test_empty_state_is_zero(self):
        self.assertEqual(protocol.estimate_tokens(""), 0)

    def test_english_is_about_a_quarter_of_the_byte_count(self):
        text = "a" * 4_000
        self.assertEqual(protocol.estimate_tokens(text, 4.0), 1_000)

    def test_non_latin_is_never_under_estimated_by_character_count(self):
        """CJK costs about one token per character; bytes/4 must not come in below that."""
        cjk = "決" * 1_000                      # 3 bytes each in UTF-8
        estimate = protocol.estimate_tokens(cjk, 4.0)
        self.assertEqual(estimate, 750)          # 3000 bytes / 4
        self.assertGreater(estimate, len(cjk) * 0.5)

    def test_estimate_is_never_zero_for_a_non_empty_state(self):
        self.assertGreaterEqual(protocol.estimate_tokens("hi", 4.0), 1)

    def test_object_state_is_encoded_as_json(self):
        self.assertEqual(protocol.encode_state({"a": 1}), '{"a":1}')
        self.assertEqual(protocol.encode_state("plain"), "plain")


class QuestionShapeTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def _bad(self, name, question, status=422):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.check_question(name, question, self.s)
        self.assertEqual(caught.exception.status, status)
        return caught.exception.detail

    def test_choice_is_accepted(self):
        out = protocol.check_question("k", {"type": "choice", "instructions": "Which?",
                                            "criteria": {"a": "first", "b": "second"}}, self.s)
        self.assertEqual(out["criteria"], {"a": "first", "b": "second"})

    def test_score_is_accepted(self):
        out = protocol.check_question("k", {"type": "score", "instructions": "How much?",
                                            "criteria": ["low", "mid", "high"]}, self.s)
        self.assertEqual(out["criteria"], ["low", "mid", "high"])

    def test_noul_is_accepted_without_criteria(self):
        out = protocol.check_question("k", {"type": "noul", "instructions": "Is it risky?"}, self.s)
        self.assertNotIn("criteria", out)

    def test_noul_accepts_the_true_false_pair(self):
        out = protocol.check_question("k", {"type": "noul", "instructions": "Risky?",
                                            "criteria": {"true": "yes", "false": "no"}}, self.s)
        self.assertEqual(set(out["criteria"]), {"true", "false"})

    def test_unknown_type_is_refused(self):
        self._bad("k", {"type": "ranking", "instructions": "x"})

    def test_missing_type_is_refused(self):
        self.assertIn("has no type", self._bad("k", {"instructions": "x"}))

    def test_empty_instructions_are_refused(self):
        self._bad("k", {"type": "noul", "instructions": "   "})

    def test_one_option_is_not_a_choice(self):
        self._bad("k", {"type": "choice", "instructions": "x", "criteria": {"only": "one"}})

    def test_a_large_choice_is_refused_with_413_not_422(self):
        """413 is what a client must see for 'too wide', so it fails open rather than retries."""
        criteria = {f"opt{i}": f"option {i}" for i in range(self.s.max_choice_options + 1)}
        detail = self._bad("k", {"type": "choice", "instructions": "x", "criteria": criteria}, 413)
        self.assertIn("Shortlist first", detail)

    def test_the_largest_allowed_choice_is_accepted(self):
        criteria = {f"opt{i}": f"option {i}" for i in range(self.s.max_choice_options)}
        out = protocol.check_question("k", {"type": "choice", "instructions": "x", "criteria": criteria}, self.s)
        self.assertEqual(len(out["criteria"]), self.s.max_choice_options)

    def test_score_over_the_level_cap_is_refused(self):
        levels = [f"level {i}" for i in range(self.s.max_score_levels + 1)]
        self._bad("k", {"type": "score", "instructions": "x", "criteria": levels}, 413)

    def test_empty_score_level_is_refused(self):
        self._bad("k", {"type": "score", "instructions": "x", "criteria": ["ok", ""]})

    def test_noul_criteria_other_than_true_false_are_refused(self):
        self._bad("k", {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}})

    def test_structured_instructions_are_accepted(self):
        out = protocol.check_question("k", {"type": "noul", "instructions": {"question": "risky?",
                                                                            "field": "risk"}}, self.s)
        self.assertEqual(out["type"], "noul")


class RequestShapeTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def _bad(self, body, status):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.check_request(body, self.s)
        self.assertEqual(caught.exception.status, status)
        return caught.exception.detail

    def test_a_minimal_request_is_accepted(self):
        out = protocol.check_request({"state": "hello",
                                      "questions": {"q": {"type": "noul", "instructions": "ok?"}}}, self.s)
        self.assertEqual(out["state"], "hello")
        self.assertIn("_encoded", out)

    def test_state_is_required(self):
        self.assertIn("state is required", self._bad({"questions": {"q": {"type": "noul", "instructions": "x"}}}, 400))

    def test_null_state_is_refused(self):
        self._bad({"state": None, "questions": {"q": {"type": "noul", "instructions": "x"}}}, 400)

    def test_oversize_state_is_refused(self):
        self._bad({"state": "x" * (self.s.max_state_chars + 1),
                   "questions": {"q": {"type": "noul", "instructions": "x"}}}, 413)

    def test_questions_are_required(self):
        self._bad({"state": "hello"}, 400)

    def test_too_many_questions_are_refused(self):
        questions = {f"q{i}": {"type": "noul", "instructions": f"question {i}"}
                     for i in range(self.s.max_questions + 1)}
        self._bad({"state": "hello", "questions": questions}, 413)

    def test_hook_arguments_are_refused_not_ignored(self):
        detail = self._bad({"state": "hello", "hooks": ["x"],
                            "questions": {"q": {"type": "noul", "instructions": "x"}}}, 422)
        self.assertIn("cannot be sent here", detail)

    def test_max_len_over_the_budget_is_refused(self):
        self._bad({"state": "hello", "max_len": self.s.max_token_budget + 1,
                   "questions": {"q": {"type": "noul", "instructions": "x"}}}, 413)

    def test_max_len_must_be_a_positive_integer(self):
        self._bad({"state": "hello", "max_len": 0,
                   "questions": {"q": {"type": "noul", "instructions": "x"}}}, 422)
        self._bad({"state": "hello", "max_len": "big",
                   "questions": {"q": {"type": "noul", "instructions": "x"}}}, 422)

    def test_min_confidence_outside_the_unit_interval_is_refused(self):
        self._bad({"state": "hello", "min_confidence": 1.5,
                   "questions": {"q": {"type": "noul", "instructions": "x"}}}, 422)

    def test_a_jev_style_model_id_is_passed_through(self):
        out = protocol.check_request({"state": "hello", "model": "jev-latest",
                                      "questions": {"q": {"type": "noul", "instructions": "x"}}}, self.s)
        self.assertEqual(out["model"], "jev-latest")

    def test_body_that_is_not_an_object_is_refused(self):
        self._bad(["not", "an", "object"], 400)


class ConfidenceTests(unittest.TestCase):
    """The metric swap that decides whether a client's thresholds ever fire."""

    def test_jevs_formula(self):
        # (n*p_max - 1)/(n - 1)
        self.assertAlmostEqual(protocol.jev_confidence([0.7, 0.1, 0.1, 0.1]), 0.6, places=6)
        self.assertAlmostEqual(protocol.jev_confidence([0.25] * 4), 0.0, places=6)
        self.assertAlmostEqual(protocol.jev_confidence([1.0, 0.0, 0.0, 0.0]), 1.0, places=6)
        self.assertAlmostEqual(protocol.jev_confidence([0.5, 0.5]), 0.0, places=6)

    def test_a_confident_answer_clears_a_jev_tuned_threshold_where_entropy_does_not(self):
        """The empirical finding: same distribution, ~0.25 entropy vs 0.6 Jev."""
        probabilities = [0.7, 0.1, 0.1, 0.1]
        k = len(probabilities)
        import math

        entropy = 1 - (-sum(p * math.log(p) for p in probabilities if p) / math.log(k))
        self.assertLess(entropy, 0.35)
        self.assertGreater(protocol.jev_confidence(probabilities), 0.59)

    def test_recast_keeps_the_engines_own_value(self):
        answer = {"type": "choice", "choice": "a",
                  "probabilities": {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1},
                  "confidence": 0.25, "answer_confidence": 0.7}
        recast = protocol.recast_confidence({"k": answer})["k"]
        self.assertAlmostEqual(recast["confidence"], 0.6, places=3)
        self.assertEqual(recast["confidence_laya"], 0.25)
        self.assertEqual(recast["answer_confidence"], 0.7)   # untouched
        self.assertEqual(recast["probabilities"], answer["probabilities"])

    def test_the_laya_style_passes_everything_through(self):
        answer = {"type": "score", "score": 1.5, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
                  "confidence": 0.31}
        recast = protocol.recast_confidence({"k": answer}, "laya")["k"]
        self.assertEqual(recast["confidence"], 0.31)
        self.assertNotIn("confidence_laya", recast)

    def test_a_noul_is_left_alone(self):
        """Two outcomes: Laya's max(p_yes, 1-p_yes) is already the gateable number."""
        answer = {"type": "noul", "noul": 0.9, "confidence": 0.9}
        recast = protocol.recast_confidence({"k": answer})["k"]
        self.assertEqual(recast["confidence"], 0.9)
        self.assertNotIn("confidence_laya", recast)

    def test_recast_does_not_mutate_the_original(self):
        answer = {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2},
                  "confidence": 0.1}
        source = {"k": answer}
        protocol.recast_confidence(source)
        self.assertEqual(source["k"]["confidence"], 0.1)


class EnvelopeTests(unittest.TestCase):
    def test_the_three_keys_a_jev_client_reads_are_present(self):
        reply = protocol.envelope({"q": {"type": "noul", "noul": 0.5}},
                                  {"input_tokens": 10}, {"model": "english"},
                                  bridge={"mode": "single"})
        for key in ("model", "answers", "usage", "routing"):
            self.assertIn(key, reply)
        self.assertEqual(reply["answers"]["q"]["noul"], 0.5)
        self.assertEqual(reply["laya"]["mode"], "single")

    def test_an_empty_envelope_is_still_well_formed(self):
        reply = protocol.envelope({}, None, None)
        self.assertEqual(reply["answers"], {})
        self.assertEqual(reply["usage"], {})


if __name__ == "__main__":
    unittest.main()
