"""The decision engine: a Laya ``Router`` plus the budget policy that makes it safe.

Why this module exists
----------------------
Laya's ``predict`` scores a state from a *single* window and silently cuts off
everything past ``max_len`` -- which defaults to 512 tokens and cannot be raised
by an environment variable in the stock server. An agent-side client does not
send ``max_len`` at all, because the cloud engine it was written against reads
the whole state. The bad outcome is not an error: it is a well-formed answer
computed from the first half of the input, which passes every schema check and
is therefore acted on.

So every request here goes through a deliberate choice:

``single``
    The state fits one comfortable pass. ``max_len`` is raised to fit it exactly,
    so nothing is cut, and the model answers in one forward pass.

``multilingual``
    The state is too big for the small checkpoint's window but fits the long one.
    ``laya-multilingual`` reads up to 8,192 tokens, so this is still a single
    forward pass -- and it is the right answer rather than a scan. Measured on CPU:
    a 7,700-token state answered this way takes seconds; the same state scanned
    window-by-window took **182 seconds** over 42 windows, because the English
    checkpoint's window is 512 tokens and the scan pays for every one of them.

``scan``
    Larger than any single window. The state is scored window by window with
    ``Router.predict_long`` and aggregated per question (``noul`` takes the
    strongest window, ``choice`` and ``score`` the most confident one). Nothing is
    discarded -- but it is slow, so it is bounded: a state whose scan would exceed
    ``LFA_SCAN_MAX_TOKENS`` is refused, and the client fails open immediately
    instead of holding a decision for minutes.

Every mode is reported in the reply, and Laya's own ``truncated`` /
``state_tokens_dropped`` usage fields are surfaced alongside it, so a state that
was cut is visible rather than silent.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Mapping, Optional

from . import protocol
from .settings import Settings

_log = logging.getLogger("laya_for_agents.engine")

# Laya raises plain ValueError for a question it cannot encode, including one
# whose option markers do not all fit in the token budget, and for a checkpoint or
# workflow name it does not know. Every one of those is the caller's mistake and
# should read as 422 -- which a Jev client fails open on -- rather than as a 500.
_QUESTION_ERROR_HINTS = (
    "option markers fit",
    "option budget",
    "max_len",
    "head_max_len",
    "option",
    "criteria",
    "question",
    "unknown model",
    "unknown task",
    "unknown workflow",
)


class Busy(RuntimeError):
    """More requests are in flight than this deployment admits."""


class Engine:
    """Owns one Laya ``Router`` and answers Jev-shaped requests against it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._router: Any = None
        self._lock = threading.Lock()
        self._admission = threading.Semaphore(settings.max_concurrent)
        # One forward pass at a time: the shape a single checkpoint on one device
        # wants. Admission is a separate, wider gate, exactly as Laya's own
        # server separates the two.
        self._inference = threading.Lock()
        self.stats: Dict[str, Any] = {
            "requests": 0, "single": 0, "multilingual": 0, "scan": 0, "errors": 0, "truncated": 0,
            "tokens_scanned": 0, "last_mode": None, "last_latency_ms": None,
            "started": time.time(),
        }

    # ── lifecycle ────────────────────────────────────────────────────────────

    def router(self) -> Any:
        """The ``Router``, built on first use so import stays free of torch."""
        if self._router is None:
            with self._lock:
                if self._router is None:
                    self._router = self._build()
        return self._router

    def _build(self) -> Any:
        from laya import Router

        kwargs: Dict[str, Any] = {"preload": False}   # warm() decides, once _build has returned
        if self.settings.device:
            kwargs["device"] = self.settings.device
        if self.settings.max_loaded:
            kwargs["max_loaded"] = self.settings.max_loaded
        # NOTE: `models` is deliberately not set from LFA_MODELS. Router's `models` is a
        # mapping of name -> (repo, subfolder), and building one from a bare name list
        # would make it try to load a repo literally called "english". Which checkpoints
        # are resident is a matter of what gets preloaded, and `Router.preload(names)`
        # takes plain names, so LFA_MODELS is applied there instead -- see warm().
        router = Router(**kwargs)
        if self.settings.threads:
            try:  # CPU inference only; oversubscribing logical cores is a large regression
                import torch

                torch.set_num_threads(int(self.settings.threads))
            except Exception:  # noqa: BLE001 - a thread hint must never stop startup
                pass
        _log.info("Laya Router built (device=%s, preload=%s)",
                  self.settings.device or "auto",
                  ",".join(self.settings.models) or "all")
        return router

    def warm(self) -> Dict[str, Any]:
        """Load the checkpoints and return what is resident.

        The load happens here rather than inside ``_build`` so an incremental
        preload does not fight the constructor, and so a failure to preload leaves a
        working server that simply loads on first use.
        """
        router = self.router()
        if self.settings.preload:
            try:
                # Names, not a repo mapping: `Router.preload` normalises its arguments.
                # None means the Router's own default set.
                router.preload(list(self.settings.models) or None)
            except Exception as error:  # noqa: BLE001
                _log.warning("preload failed (checkpoints will load on first use): %s", error)
        return self.describe()

    # ── introspection ────────────────────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"loaded": [], "loaded_revisions": {}, "device": self.settings.device or "auto"}
        if self._router is None:
            return out
        # On Laya's Router both of these are properties, not methods. Reading them
        # defensively means a version that turns them back into methods, or an
        # injected stand-in that has them as methods, still reports correctly
        # instead of silently describing an empty server.
        out["loaded"] = self._read(self._router, "loaded", [])
        out["loaded_revisions"] = self._read(self._router, "loaded_revisions", {})
        out["device"] = self._device_report()
        return out

    @staticmethod
    def _read(target: Any, name: str, default: Any) -> Any:
        try:
            value = getattr(target, name)
            value = value() if callable(value) else value
            return list(value) if isinstance(default, list) else dict(value)
        except Exception:  # noqa: BLE001 - introspection must never break /health
            return default

    def _device_report(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
            return "cpu"
        except Exception:  # noqa: BLE001
            return self.settings.device or "auto"

    # ── budget policy ────────────────────────────────────────────────────────

    def plan(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Decide how this state will be scored, before any model work happens.

        Separated from :meth:`answer` so the policy is inspectable and testable
        without loading a checkpoint -- and so a caller can see why a request was
        scanned rather than answered in one pass.
        """
        encoded = request["_encoded"]
        estimated = protocol.estimate_tokens(encoded, self.settings.tokens_per_char_divisor)
        single_max = int(getattr(self.settings, "single_max", self.settings.default_max_len))
        # The state budget must cover the state and leave room for the question.
        needed = estimated + self._question_head_room(request["questions"])
        if needed <= single_max:
            mode = "single"
        elif (self.settings.long_policy == "multilingual"
              and needed <= self.settings.multilingual_budget):
            # One pass on the long checkpoint beats a window-by-window scan of the
            # short one by two orders of magnitude on CPU. Measured: 42 windows of
            # the 512-token English checkpoint took 182 s for a 7.7k-token state.
            mode = "multilingual"
        elif estimated <= self.settings.scan_max_tokens:
            mode = "scan"
        else:
            # A scan this large holds the request for minutes, which is worse than no
            # decision at all for a caller on a 2.5 s budget -- the decision arrives
            # after the turn it was for has already been answered by the fallback.
            # Refusing lets the client fail open immediately and say so.
            raise protocol.ProtocolError(
                413, f"state is about {estimated} tokens, past the {self.settings.multilingual_budget}-token "
                     f"long checkpoint and past this build's {self.settings.scan_max_tokens}-token scan "
                     f"budget. A scan of this size takes minutes on CPU and this server will not hold a "
                     f"turn for it. Trim the state, or set LFA_LONG_POLICY=scan with a higher "
                     f"LFA_SCAN_MAX_TOKENS and accept the latency")
        return {
            "mode": mode,
            "estimated_state_tokens": estimated,
            "needed_tokens": needed,
            "single_max": single_max,
            "multilingual_budget": self.settings.multilingual_budget,
            "state_chars": len(encoded),
        }

    def _question_head_room(self, questions: Mapping[str, Any]) -> int:
        """Tokens the question side of the sequence will need.

        The state and the questions share one ``max_len`` window, and Laya spends
        ``head_max_len`` of it on the option markers. Estimating that here is what
        keeps a wide question from crowding out the state (or the state from
        crowding out the options, which Laya reports as a ValueError).

        The text is only part of it: Laya wraps every option in marker tokens, so a
        question with twenty options costs more than its characters suggest. That
        per-option cost is counted separately -- it is what decides whether a wide
        question is answered or refused.
        """
        chars = 0
        options = 0
        for question in questions.values():
            chars += len(str(question.get("instructions", "")))
            criteria = question.get("criteria")
            if isinstance(criteria, Mapping):
                options += len(criteria)
                chars += sum(len(str(k)) + len(str(v)) for k, v in criteria.items())
            elif isinstance(criteria, (list, tuple)):
                options += len(criteria)
                chars += sum(len(str(v)) for v in criteria)
        text = protocol.estimate_tokens("x" * chars, self.settings.tokens_per_char_divisor)
        return max(32, text + options * self.settings.marker_tokens + 8)

    @staticmethod
    def _resolve_model(model: Optional[str]) -> Optional[str]:
        """Map a caller's ``model`` onto a Laya checkpoint, or ``None`` to auto-select.

        This mirrors what ``laya-serve`` does in ``_resolve_model``, and it is not
        optional politeness -- ``Router.predict`` passes the value straight to
        ``normalise_name``, which **raises** on anything it does not recognise. A Jev
        client always sends ``model: "jev-latest"`` (and OpenCode Zen's free tier sends
        ``jev-1.13-free``), so without this every single request from a real Jev client
        is a 500. Both Laya's server and Jev's own docs agree on the intent: an id that
        is not a checkpoint means "let the router choose".

        Published Hugging Face ids are accepted too, the way ``laya-serve`` accepts
        them, and the checkpoints are read out of Laya's own registry rather than
        copied here so one added upstream is recognised without a change.
        """
        if not model:
            return None
        wanted = str(model).strip().lower()
        try:
            from laya.router import BUNDLE_REPO, STANDALONE_MODELS

            published = {repo.lower(): name for name, repo in STANDALONE_MODELS.items()
                         if repo != BUNDLE_REPO}
            if wanted in published:
                return published[wanted]
        except Exception:  # noqa: BLE001 - fall through to the name check
            pass
        try:
            from laya.router import normalise_name

            return normalise_name(str(model))
        except Exception:  # noqa: BLE001 - an unrecognised id is exactly the auto-select case
            return None

    def _head_budget(self, request: Mapping[str, Any]) -> int:
        """``head_max_len``, widened when a question's options need the room.

        The stock default of 192 tokens is spent entirely on the option markers,
        and a question whose markers overrun it is refused rather than answered.
        Widening it here turns that refusal into an answer, bounded by
        ``LFA_HEAD_MAX_LEN_CEILING`` so one wide question cannot starve the state.
        """
        asked = request.get("head_max_len")
        if asked:
            return int(asked)
        ceiling = int(getattr(self.settings, "head_max_len_ceiling", 512))
        needed = self._question_head_room(request["questions"])
        return max(self.settings.default_head_max_len, min(ceiling, needed))

    # ── answering ────────────────────────────────────────────────────────────

    def answer(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Score one validated request. Raises ``Busy`` or ``protocol.ProtocolError``."""
        if not self._admission.acquire(blocking=False):
            raise Busy("server busy, try again later")
        try:
            started = time.monotonic()
            plan = self.plan(request)
            state = request["state"]
            questions = request["questions"]
            # A caller's checkpoint choice, resolved. An id Laya does not know (a Jev id
            # like `jev-latest`) resolves to None and is dropped rather than forwarded,
            # because forwarding it makes Router.predict raise. See _resolve_model.
            named = self._resolve_model(request.get("model"))
            common: Dict[str, Any] = {}
            if named:
                common["model"] = named
            for key in ("task", "lang", "lang_guess"):
                if request.get(key):
                    common[key] = request[key]

            router = self.router()
            with self._inference:
                if plan["mode"] == "scan":
                    result = router.predict_long(
                        state, questions, **common,
                        window=self.settings.long_window,
                        stride=self.settings.long_stride,
                        aggregate=self.settings.long_aggregate,
                        batch_size=self.settings.long_batch_size,
                    )
                else:
                    # The mode decision already guaranteed `needed <= single_max`, and a
                    # caller's explicit budget was validated against max_token_budget. So
                    # the window is simply whatever covers the request -- never the stock
                    # 512, which is what would silently cut the state. The estimate is an
                    # upper bound, and `budget_slack` covers the marker and template tokens
                    # it does not model; the floor keeps a tiny state on the same window the
                    # stock server would have given it, so nothing changes where nothing
                    # needed to.
                    asked = int(request.get("max_len") or 0)
                    max_len = max(asked,
                                  plan["needed_tokens"] + self.settings.budget_slack,
                                  self.settings.window_floor)
                    max_len = min(max_len, self.settings.max_token_budget)
                    if plan["mode"] == "multilingual" and not named:
                        # Pin the long checkpoint -- but only when the caller did not name
                        # one. Routing would otherwise send English text to the 512-token
                        # checkpoint, which is the truncation this package exists to
                        # prevent. A Jev client always sends `model: "jev-latest"`, which
                        # resolves to None above, so it does not count as naming one. A
                        # caller who really does name `english` keeps it, and the reply
                        # says `truncated: true` if it then cuts the state.
                        common["model"] = "multilingual"
                    result = router.predict(
                        state, questions, **common,
                        max_len=max_len, head_max_len=self._head_budget(request),
                        min_confidence=request.get("min_confidence"),
                    )

            latency = int((time.monotonic() - started) * 1000)
            self.stats["requests"] += 1
            self.stats[plan["mode"]] += 1
            self.stats["last_mode"] = plan["mode"]
            self.stats["last_latency_ms"] = latency
            self.stats["tokens_scanned"] += plan["estimated_state_tokens"]

            usage = dict(result.get("usage") or {})
            usage.setdefault("input_tokens", plan["estimated_state_tokens"])
            usage.setdefault("output_tokens", 0)
            # Laya reports its own truncation status. Surfacing it is the whole point
            # of this package: a state that was cut must be visible, not inferred from
            # an answer that simply looks a bit off.
            truncated = bool(usage.get("truncated"))
            if truncated:
                self.stats["truncated"] += 1
            bridge = {
                "mode": plan["mode"],
                "estimated_state_tokens": plan["estimated_state_tokens"],
                "state_chars": plan["state_chars"],
                "single_max": plan["single_max"],
                "latency_ms": latency,
                "truncated": truncated,
                "state_tokens_dropped": int(usage.get("state_tokens_dropped") or 0),
                "note": {
                    "scan": "scanned every window of the state; nothing truncated",
                    "multilingual": "answered in one pass on the long checkpoint",
                }.get(plan["mode"], "answered in one pass with the state budget raised to fit"),
            }
            if isinstance(result.get("usage", {}).get("windows"), int):
                bridge["windows"] = result["usage"]["windows"]
            bridge["confidence_style"] = self.settings.confidence_style
            # The client's thresholds assume Jev's confidence. See protocol.jev_confidence.
            answers = protocol.recast_confidence(result.get("answers") or {},
                                                 self.settings.confidence_style)
            return protocol.envelope(answers, usage, result.get("routing") or {}, bridge=bridge)
        except Busy:
            raise
        except protocol.ProtocolError:
            raise
        except Exception as error:  # noqa: BLE001
            self.stats["errors"] += 1
            message = str(error)
            lowered = message.lower()
            if any(hint in lowered for hint in _QUESTION_ERROR_HINTS):
                # A question the checkpoint cannot encode: the caller's mistake, and
                # one a Jev client should fail open on rather than retry.
                raise protocol.ProtocolError(422, message) from None
            if "out of memory" in lowered or "cuda oom" in lowered:
                raise Busy("server busy, try again later") from None
            _log.exception("inference failed")
            # A fixed string, so paths, weights and memory state never leak to a caller.
            raise protocol.ProtocolError(500, "inference failed") from None
        finally:
            self._admission.release()

    def health(self) -> Dict[str, Any]:
        """The ``/health`` body: what is resident, what it runs on, what it has done."""
        described = self.describe()
        return {
            "status": "ok",
            "service": "laya-for-agents",
            "loaded": described["loaded"],
            "revisions": described["loaded_revisions"],
            "device": described["device"],
            "protocol": "Jev /v1/systemone",
            "budget": {
                "single_max": int(getattr(self.settings, "single_max", self.settings.default_max_len)),
                "default_max_len": self.settings.default_max_len,
                "head_max_len": self.settings.default_head_max_len,
                "head_max_len_ceiling": int(getattr(self.settings, "head_max_len_ceiling", 512)),
                "max_token_budget": self.settings.max_token_budget,
                "auto_long": self.settings.auto_long,
                "long_policy": self.settings.long_policy,
                "multilingual_budget": self.settings.multilingual_budget,
                "scan_max_tokens": self.settings.scan_max_tokens,
                "confidence_style": self.settings.confidence_style,
                "long_window": self.settings.long_window or "checkpoint default",
            },
            "stats": {k: v for k, v in self.stats.items() if k != "started"},
            "uptime_s": int(time.time() - self.stats["started"]),
        }
