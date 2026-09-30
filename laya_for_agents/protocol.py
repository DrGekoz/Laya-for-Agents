"""The Jev ``/v1/systemone`` wire protocol: validation, estimation, envelopes.

TypeSafe Jev and Laya speak the same three question types -- ``choice`` (one of a
closed set), ``score`` (a position on an ordered rubric) and ``noul`` (the
probability of a yes). A client written against Jev (here: ``hermes-jev-skills``)
sends a ``state`` plus a mapping of named questions and reads back an ``answers``
mapping. Nothing in this module writes prose; it only shapes and checks.

Validation lives here rather than being delegated so that a bad request produces
a precise HTTP status instead of a silently odd decision. A Jev client treats
every non-200 as "the decision engine had no opinion" and takes its fail-open
path, which is exactly the right outcome for a request we cannot answer
correctly.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

QUESTION_TYPES = ("choice", "score", "noul")
NOUL_CRITERIA_KEYS = ("true", "false")


class ProtocolError(Exception):
    """A request this server will not answer, with the status it deserves."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ── state ────────────────────────────────────────────────────────────────────

def encode_state(state: Any) -> str:
    """The text a model would actually be given, as one string.

    A string state is itself; an object or array is its JSON. ``ensure_ascii``
    stays off so the byte count below reflects what is really sent, and the
    separator is compact so a large mapping does not pay for pretty printing.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False, default=str)


def estimate_tokens(encoded: str, divisor: float = 4.0) -> int:
    """A cheap, deliberately conservative upper bound on the token count.

    Used only to decide between one window and a scan, and it must never
    under-estimate -- under-estimating is what lets a state be silently cut. So
    this counts UTF-8 bytes rather than characters: a non-Latin character
    encodes to three bytes and costs about one token, where a Latin character is
    one byte and costs about a quarter of a token. Dividing bytes by four is
    therefore pessimistic for CJK and about right for English, which is the safe
    direction on both.
    """
    if not encoded:
        return 0
    by_bytes = len(encoded.encode("utf-8", errors="replace")) / max(divisor, 1.0)
    return int(math.ceil(by_bytes))


# ── questions ────────────────────────────────────────────────────────────────

def _structured(value: Any) -> bool:
    """A string with words in it, or a non-empty object or array."""
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (Mapping, list, tuple)) and bool(value)


def _shown(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


def check_question(name: str, question: Any, settings: Any) -> Dict[str, Any]:
    """One question in the shape both Jev and Laya answer, or a ``ProtocolError``.

    The caps are the ones Laya's own HTTP server enforces. They are repeated here
    rather than imported from ``laya.serve`` so this bridge keeps working against
    any Laya version, and so the numbers can be raised (a widened
    ``head_max_len`` buys more options) without patching the library.
    """
    if not isinstance(question, Mapping):
        raise ProtocolError(422, f'question "{_shown(name)}" must be an object with a "type"')
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        found = "has no type" if kind is None else f'has unknown type {_shown(json.dumps(kind, default=str))}'
        raise ProtocolError(422, f'question "{_shown(name)}" {found}; use one of {", ".join(QUESTION_TYPES)}')
    instructions = question.get("instructions")
    if not _structured(instructions):
        raise ProtocolError(422, f'question "{_shown(name)}" has no instructions')
    criteria = question.get("criteria")

    if kind == "noul":
        if criteria is None:
            return {"type": "noul", "instructions": instructions}
        if (not isinstance(criteria, Mapping) or not criteria
                or not set(map(str, criteria)) <= set(NOUL_CRITERIA_KEYS)):
            raise ProtocolError(
                422, f'question "{_shown(name)}" is a noul: its criteria, if given, must be '
                     f'{{"true": ..., "false": ...}}')
        return {"type": "noul", "instructions": instructions,
                "criteria": {str(k): v for k, v in criteria.items()}}

    if kind == "choice":
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise ProtocolError(422, f'question "{_shown(name)}" is a choice: criteria must be an '
                                     f'object of 2 to {settings.max_choice_options} options')
        if len(criteria) > settings.max_choice_options:
            raise ProtocolError(
                413, f'question "{_shown(name)}" offers {len(criteria)} options; this build answers '
                     f'at most {settings.max_choice_options} per choice. Shortlist first, then choose')
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    # score
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)) or len(criteria) < 2:
        raise ProtocolError(422, f'question "{_shown(name)}" is a score: criteria must be a list of '
                                 f'2 to {settings.max_score_levels} levels, lowest first')
    if len(criteria) > settings.max_score_levels:
        raise ProtocolError(413, f'question "{_shown(name)}" has {len(criteria)} levels; this build '
                                 f'answers at most {settings.max_score_levels}')
    for position, level in enumerate(criteria):
        if not _structured(level):
            raise ProtocolError(422, f'question "{_shown(name)}": score level {position} is empty')
    return {"type": "score", "instructions": instructions, "criteria": list(criteria)}


def check_questions(questions: Any, settings: Any) -> Dict[str, Dict[str, Any]]:
    """Every question named and shaped, in the order and under the names given."""
    if not isinstance(questions, Mapping) or not questions:
        raise ProtocolError(400, "questions must be a non-empty object of {name: question}")
    if len(questions) > settings.max_questions:
        raise ProtocolError(413, f"{len(questions)} questions; this build answers at most "
                                 f"{settings.max_questions} per request")
    checked = {str(name): check_question(str(name), q, settings) for name, q in questions.items()}
    total = sum(len(q["criteria"]) for q in checked.values() if isinstance(q.get("criteria"), (Mapping, list)))
    if total > settings.max_total_options:
        raise ProtocolError(413, f"{total} options across all questions; this build accepts at most "
                                 f"{settings.max_total_options}")
    return checked


# ── request ──────────────────────────────────────────────────────────────────

_STRING_CONTROLS = ("lang", "lang_guess", "task")
_HOOK_CONTROLS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")


def check_request(body: Any, settings: Any) -> Dict[str, Any]:
    """A whole Jev request, validated, with the controls this server honours.

    Hook arguments are refused rather than ignored: a hook is a callable that
    runs inside this process, so no value a caller sends can have a meaning here.
    Silently dropping one would tell the caller its request was honoured when it
    was not.
    """
    if not isinstance(body, Mapping):
        raise ProtocolError(400, "the request body must be a JSON object")
    given_hooks = sorted(k for k in _HOOK_CONTROLS if body.get(k) is not None)
    if given_hooks:
        raise ProtocolError(422, f'{", ".join(given_hooks)} run inside the server process and cannot '
                                 f'be sent here; install them where the bridge runs, or drop them')
    if body.get("state") is None:
        raise ProtocolError(400, "state is required and must not be null")
    encoded = encode_state(body["state"])
    if len(encoded) > settings.max_state_chars:
        raise ProtocolError(413, f"state is {len(encoded)} characters; this build accepts at most "
                                 f"{settings.max_state_chars}")
    questions = check_questions(body.get("questions"), settings)

    out: Dict[str, Any] = {"state": body["state"], "questions": questions}
    for key in _STRING_CONTROLS:
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ProtocolError(422, f'{key} must be a string, or null')
        if value.strip():
            out[key] = value
    model = body.get("model")
    if isinstance(model, str) and model.strip():
        out["model"] = model.strip()

    for key, ceiling in (("max_len", settings.max_token_budget),
                         ("head_max_len", settings.max_token_budget)):
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ProtocolError(422, f"{key} must be a positive integer")
        if value > ceiling:
            raise ProtocolError(413, f"{key} is {value}; this build caps it at {ceiling}")
        out[key] = value

    min_confidence = body.get("min_confidence")
    if min_confidence is not None:
        if (not isinstance(min_confidence, (int, float)) or isinstance(min_confidence, bool)
                or not math.isfinite(float(min_confidence))
                or not 0.0 <= float(min_confidence) <= 1.0):
            raise ProtocolError(422, "min_confidence must be a number in [0, 1], or null")
        out["min_confidence"] = float(min_confidence)

    out["_encoded"] = encoded
    return out


# ── confidence ───────────────────────────────────────────────────────────────

def jev_confidence(probabilities: Sequence[float]) -> float:
    """Jev's ``confidence`` for a distribution: ``(n*p_max - 1) / (n - 1)``.

    The two engines report a **different quantity under the same name**, and it
    matters because an agent client gates on it.

    * Jev defines confidence as ``(n*p_max - 1)/(n - 1)``: the margin over a uniform
      guess, normalised. On a 4-option question answered with p_max = 0.7 that is
      ``(2.8 - 1)/3 = 0.6`` -- comfortably past a 0.6 threshold.
    * Laya reports **normalized entropy** ``1 - H(p)/log(k)`` on ``choice`` and
      ``score``. The same distribution scores about ``0.25``.

    Same answer, same probabilities, same calibration -- and the client reads one as
    "confident" and the other as "no opinion". Every routing decision then falls
    through to the fail-open path and the feature silently never fires. This is
    Laya's own documented warning ("never compare the two against one threshold"),
    made concrete.

    So a Jev client is served the metric it was written against, computed from the
    probabilities Laya actually produced. The value Laya itself reported is kept
    beside it as ``confidence_laya``, and ``LFA_CONFIDENCE_STYLE=laya`` passes it
    through untouched for anyone who would rather gate on entropy directly.
    """
    values = [float(v) for v in probabilities]
    k = len(values)
    if k < 2:
        return 1.0
    top = max(values)
    return max(0.0, min(1.0, (k * top - 1.0) / (k - 1.0)))


def recast_confidence(answers: Mapping[str, Any], style: str = "jev") -> Dict[str, Any]:
    """Rewrite each answer's ``confidence`` into the style the caller expects.

    ``choice`` and ``score`` carry a full distribution, so Jev's formula applies
    directly. A ``noul`` has two outcomes and Laya already reports ``max(p_yes,
    1-p_yes)`` there, which is the quantity a yes/no gate wants -- so it is left
    alone under either style. The engine's own value is always preserved as
    ``confidence_laya``, so nothing is lost either way.
    """
    if style != "jev":
        return {name: dict(answer) for name, answer in answers.items()}
    out: Dict[str, Any] = {}
    for name, answer in answers.items():
        recast = dict(answer)
        probabilities = answer.get("probabilities")
        if (answer.get("type") in ("choice", "score")
                and isinstance(probabilities, Mapping) and probabilities):
            if recast.get("confidence") is not None:
                recast["confidence_laya"] = recast["confidence"]
            recast["confidence"] = round(jev_confidence(list(probabilities.values())), 4)
        out[name] = recast
    return out


def envelope(answers: Mapping[str, Any], usage: Mapping[str, Any], routing: Mapping[str, Any],
             *, model: str = "laya-for-agents", bridge: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """The reply a Jev client decodes, plus a ``laya`` block of our own.

    ``answers``, ``usage`` and ``model`` are the three keys a Jev client reads;
    everything else is additive and ignored by one. The extra block is where the
    behaviour that is *not* Jev's becomes visible -- which checkpoint answered,
    whether the state was scanned rather than cut, and the budget it was given.
    """
    reply: Dict[str, Any] = {
        "model": model,
        "answers": {name: dict(answer) for name, answer in answers.items()},
        "usage": dict(usage or {}),
        "routing": dict(routing or {}),
    }
    if bridge:
        reply["laya"] = dict(bridge)
    return reply
