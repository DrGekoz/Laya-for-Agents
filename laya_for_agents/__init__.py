"""Laya for Agents: a local decision engine for agent harnesses.

TypeSafe Jev's idea is the right one -- an agent burns frontier-model tokens on
things that are not thinking (which model should answer this turn, which of the
retrieved passages are worth reading, which turns survive a summary, which
button comes next). Those are decisions, not prose, and a small model answers
them faster and cheaper with a calibrated confidence attached.

This package runs that idea on your own hardware. It serves the Jev
``/v1/systemone`` wire protocol over a local `Laya <https://github.com/NandhaKishorM/laya>`_
checkpoint, so any client written against Jev works unchanged -- in particular
`hermes-jev-skills <https://github.com/kerpopule/hermes-jev-skills>`_, which
gives Hermes model routing, memory filtering, compaction selection, skill
selection, triage and computer/browser action choice.

The bridge earns its keep in one place. Laya's ``predict`` scores a state from a
single window and silently discards everything past ``max_len`` -- 512 tokens by
default, and not raisable by an environment variable in the stock server. Agent
clients do not send ``max_len``, because the cloud engine they were written
against reads the whole state. So a long state does not error: it comes back as
a well-formed answer computed from its first half, which passes every schema
check and is therefore acted on. Laya for Agents makes that impossible. Every
request carries a real budget, and a state too large for one pass is scored
window by window and aggregated per question instead of being cut.

Quick start::

    pip install -e ".[serve]"
    laya-for-agents serve          # http://127.0.0.1:8000

    # then point a Jev client at it
    set TYPESAFE_BASE_URL=http://127.0.0.1:8000

See ``README.md`` for the Hermes wiring, the tuning knobs, and the limits that
are worth measuring on your own traffic before you trust a confidence gate.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "Settings", "load_settings", "Engine", "create_app"]


def __getattr__(name: str):  # PEP 562: keep torch out of `import laya_for_agents`
    if name == "Settings":
        from .settings import Settings

        return Settings
    if name == "load_settings":
        from .settings import load

        return load
    if name == "Engine":
        from .engine import Engine

        return Engine
    if name == "create_app":
        from .server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
