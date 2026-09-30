"""Environment-driven settings for Laya for Agents.

Everything is an environment variable, so one image serves a laptop dev run and a
systemd unit, and nothing has to be threaded through a config file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass
class Settings:
    """Resolved configuration. Built once at startup by :func:`load`."""

    # ── HTTP surface ─────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000
    root_path: str = ""
    api_key: str = ""            # when set, every /v1/systemone call needs it
    max_concurrent: int = 16
    log_level: str = "info"

    # ── model ────────────────────────────────────────────────────────────────
    device: Optional[str] = None
    preload: bool = True
    models: List[str] = field(default_factory=list)   # empty = let the Router decide
    max_loaded: int = 2
    threads: Optional[int] = None

    # ── the reason this bridge exists ────────────────────────────────────────
    # Laya's `predict` scores a state from a single window and silently cuts off
    # everything past `max_len` (default 512). These two settings make that
    # impossible: every request carries an explicit budget, and a state too big
    # for one window is scanned window-by-window instead of truncated.
    #     · LFA_SINGLE_MAX     : the largest state answered in one forward pass.
    #                            Past it, the state is scanned window by window
    #                            instead. Raise it to trade scan overhead for a
    #                            single pass; 1024 sits inside every shipped
    #                            checkpoint's positional range.
    #     · LFA_HEAD_MAX_LEN_CEILING : how far head_max_len may be widened to fit a
    #                            question's option markers, so a wide question is
    #                            answered instead of refused.
    default_max_len: int = 8192
    default_head_max_len: int = 192
    head_max_len_ceiling: int = 512
    single_max: int = 1024
    window_floor: int = 512               # never hand the model a tighter window than stock
    budget_slack: int = 64                # headroom over the estimate, so a generous one is still safe
    max_token_budget: int = 8192          # ceiling on a caller's own max_len
    auto_long: bool = True                # scan long states, never truncate
    # How a state too big for the short checkpoint is handled:
    #   "multilingual" -- one pass on laya-multilingual, which reads 8,192 tokens.
    #                     Measured on CPU: seconds, where the same state scanned
    #                     window-by-window took 182 s over 42 windows.
    #   "scan"         -- Router.predict_long over the short checkpoint's own
    #                     windows. Nothing is lost, but every window costs a pass.
    long_policy: str = "multilingual"
    multilingual_budget: int = 8192       # the long checkpoint's usable window
    # A scan past this size is refused rather than run. Scans cost a full forward
    # pass per window -- 42 of them took 182 s for a 7.7k-token state on CPU -- and a
    # decision that arrives minutes late is worse than one that fails open at once.
    # Defaults to the multilingual window, so by default nothing is scanned at all.
    # Raise it (with LFA_LONG_POLICY=scan) only where a slow answer beats none.
    scan_max_tokens: int = 8192
    long_window: Optional[int] = None     # None = the checkpoint's own budget
    long_stride: Optional[int] = None     # None = library default (half a window)
    long_aggregate: str = "auto"
    long_batch_size: Optional[int] = None
    scan_threshold_ratio: float = 0.9     # scan once the state fills this much of a window
    tokens_per_char_divisor: float = 4.0  # byte-based estimate for the pre-flight check
    marker_tokens: int = 4                # tokens Laya spends wrapping each option
    # Which `confidence` a reply carries. Laya reports normalized entropy on
    # choice/score; a Jev client's thresholds assume Jev's (n*p_max-1)/(n-1). Those
    # differ by roughly 2x and a client gating on 0.6 sees Laya's ~0.25 as "no
    # opinion", so every decision falls through to the fail-open path. "jev" serves
    # the metric the client was written against (the engine's own value is kept as
    # `confidence_laya`); "laya" passes the engine's value through untouched.
    confidence_style: str = "jev"

    # ── protocol caps (mirrors Laya's own HTTP guardrails) ───────────────────
    max_questions: int = 64
    max_choice_options: int = 100
    max_score_levels: int = 32
    max_total_options: int = 512
    max_state_chars: int = 50_000
    max_body_bytes: int = 2 * 1024 * 1024


def load() -> Settings:
    """Read the environment once, into a Settings object."""
    models = [m.strip() for m in _env("LFA_MODELS").split(",") if m.strip()]
    threads = _int("LFA_THREADS", 0)
    return Settings(
        host=_env("LFA_HOST", "127.0.0.1"),
        port=_int("LFA_PORT", 8000),
        root_path=_env("LFA_ROOT_PATH", ""),
        api_key=_env("LFA_API_KEY") or _env("LAYA_API_KEY"),
        max_concurrent=_int("LFA_MAX_CONCURRENT", 16),
        log_level=_env("LFA_LOG_LEVEL", "info"),
        device=_env("LFA_DEVICE") or _env("LAYA_DEVICE") or None,
        preload=_flag("LFA_PRELOAD", True),
        models=models or [m.strip() for m in _env("LAYA_MODELS").split(",") if m.strip()],
        max_loaded=_int("LFA_MAX_LOADED", 2),
        threads=threads or None,
        default_max_len=_int("LFA_MAX_LEN", 8192),
        default_head_max_len=_int("LFA_HEAD_MAX_LEN", 192),
        head_max_len_ceiling=_int("LFA_HEAD_MAX_LEN_CEILING", 512),
        single_max=_int("LFA_SINGLE_MAX", 1024),
        window_floor=_int("LFA_WINDOW_FLOOR", 512),
        budget_slack=_int("LFA_BUDGET_SLACK", 64),
        max_token_budget=_int("LFA_MAX_TOKEN_BUDGET", _int("LFA_MAX_LEN", 8192)),
        auto_long=_flag("LFA_AUTO_LONG", True),
        long_policy=_env("LFA_LONG_POLICY", "multilingual").lower(),
        multilingual_budget=_int("LFA_MULTILINGUAL_BUDGET", 8192),
        scan_max_tokens=_int("LFA_SCAN_MAX_TOKENS", _int("LFA_MULTILINGUAL_BUDGET", 8192)),
        long_window=_int("LFA_LONG_WINDOW", 0) or None,
        long_stride=_int("LFA_LONG_STRIDE", 0) or None,
        long_aggregate=_env("LFA_LONG_AGGREGATE", "auto"),
        long_batch_size=_int("LFA_LONG_BATCH", 0) or None,
        scan_threshold_ratio=float(_env("LFA_SCAN_THRESHOLD", "0.9") or 0.9),
        tokens_per_char_divisor=float(_env("LFA_TOKENS_PER_CHAR", "4.0") or 4.0),
        marker_tokens=_int("LFA_MARKER_TOKENS", 4),
        confidence_style=_env("LFA_CONFIDENCE_STYLE", "jev").lower(),
        max_questions=_int("LFA_MAX_QUESTIONS", 64),
        max_choice_options=_int("LFA_MAX_CHOICE_OPTIONS", 100),
        max_score_levels=_int("LFA_MAX_SCORE_LEVELS", 32),
        max_total_options=_int("LFA_MAX_TOTAL_OPTIONS", 512),
        max_state_chars=_int("LFA_MAX_STATE_CHARS", 50_000),
        max_body_bytes=_int("LFA_MAX_BODY_BYTES", 2 * 1024 * 1024),
    )
