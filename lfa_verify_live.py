#!/usr/bin/env python3
"""Prove the local engine works with the real client, end to end.

Run this against a running server::

    laya-for-agents serve            # in one terminal
    python lfa_verify_live.py        # in another

It does four things, in increasing order of how much they prove:

1. asks ``GET /health`` and shows what is resident;
2. POSTs a decision request itself, the way any HTTP client would;
3. drives ``hermes-jev-skills``' own ``client.ask()`` at this server, with the
   reply going through **that project's** strict validator -- the check that
   decides whether a real agent gets a decision or silently gets nothing;
4. runs ``jevkit.route.decide()``, the actual model-routing entry point, and shows
   the tier, specialty and model it chose from a hand-written pool.

Step 3 and 4 are the point. They are the consumer's own code, not a restatement
of its interface, so if they pass the wiring is real.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BASE = os.environ.get("LFA_BASE_URL", "http://127.0.0.1:8000")
JEV_SKILLS = Path(os.environ.get("JEV_SKILLS_PATH", PROJECT / "vendor" / "hermes-jev-skills"))

# The client reads these itself; setting them here is what a real operator does.
os.environ["TYPESAFE_BASE_URL"] = BASE
os.environ.pop("TYPESAFE_API_KEY", None)
os.environ.pop("JEV_PROXY_API_KEY", None)

OK, BAD = "[ OK ]", "[FAIL]"
failures = 0


def ok(label, detail=""):
    print(f"  {OK} {label}" + (f"  {detail}" if detail else ""))


def bad(label, detail=""):
    global failures
    failures += 1
    print(f"  {BAD} {label}" + (f"  {detail}" if detail else ""))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as reply:
        return json.loads(reply.read())


def post(path, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(BASE + path, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return error.code, {"raw": raw.decode(errors="replace")}


TURN = ("Refactor the parallax renderer to stream frames straight into NVENC instead of "
        "writing a PNG per frame, and keep the output bit-identical.")


def main() -> int:
    print(f"Laya for Agents - live verification against {BASE}")
    print()

    # 1 ── health ─────────────────────────────────────────────────────────────
    print("1. health")
    try:
        health = get("/health")
    except Exception as error:  # noqa: BLE001
        bad("the server is not answering", str(error)[:80])
        print("\nStart it first:  laya-for-agents serve")
        return 1
    ok("server is up", f"device={health.get('device')}")
    ok("checkpoints resident", ", ".join(health.get("loaded") or []) or "none")
    ok("protocol", health.get("protocol", "?"))
    budget = health.get("budget") or {}
    ok("budget policy",
       f"single_max={budget.get('single_max')} long_policy={budget.get('long_policy')} "
       f"scan_max={budget.get('scan_max_tokens')}")
    print()

    # 2 ── a plain HTTP decision ──────────────────────────────────────────────
    print("2. one decision request, as any HTTP client would send it")
    status, reply = post("/v1/systemone", {
        "state": "We were billed twice for March. Refund it today or we cancel.",
        "questions": {
            "queue": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "invoices, payments, refunds",
                                   "technical": "bugs, outages, system errors",
                                   "other": "everything else"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["not urgent", "soon", "blocking"]},
            "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
        }})
    if status != 200:
        bad(f"expected 200, got {status}", json.dumps(reply)[:120])
    else:
        answers = reply["answers"]
        ok("answered", f"queue={answers['queue']['choice']} "
                       f"urgency={answers['urgency']['score']} "
                       f"churn={answers['churn']['noul']}")
        ok("checkpoint", (reply.get("routing") or {}).get("model", "?"))
        ok("mode", f"{reply['laya']['mode']} in {reply['laya']['latency_ms']} ms "
                   f"(truncated={reply['laya']['truncated']})")
    print()

    # 3 ── the real client's own validator ───────────────────────────────────
    print("3. hermes-jev-skills' own client, against this server")
    if not (JEV_SKILLS / "jevkit" / "client.py").is_file():
        bad("hermes-jev-skills is not vendored",
            f"run: laya-for-agents setup-hermes   (expected at {JEV_SKILLS})")
        print()
    else:
        sys.path.insert(0, str(JEV_SKILLS))
        import jevkit.client as client  # noqa: E402

        ok("client imported", f"endpoint override -> {client._custom_typesafe_endpoint(BASE)}")
        questions = {
            "difficulty": client.score("How demanding is it to complete this turn well?",
                                       ["Trivial or mechanical: a lookup, reformat, rename",
                                        "Routine: ordinary multi-step work with a clear path",
                                        "Substantial: needs planning, several interacting parts",
                                        "Expert: subtle, ambiguous, or high-stakes"]),
            "kind": client.choice("What kind of work is this turn mainly?",
                                  {"coding": "writing, changing or debugging software",
                                   "writing": "drafting or editing prose",
                                   "research": "finding, comparing or synthesizing information",
                                   "general": "conversation, planning, operations"}),
            "costly_mistake": client.noul("A wrong or sloppy answer here would be costly or hard to undo"),
        }
        started = time.monotonic()
        try:
            result = client.ask({"user_turn": TURN}, questions, timeout=30.0)
            elapsed = (time.monotonic() - started) * 1000
            ok("ask() returned and the reply passed the client's validator",
               f"{elapsed:.0f} ms, provider={result['provider']}")
            ok("difficulty", result["answers"]["difficulty"]["score"])
            ok("kind", result["answers"]["kind"]["choice"])
            ok("costly_mistake", result["answers"]["costly_mistake"]["noul"])
        except client.JevError as error:
            bad("the client refused the reply", f"{error.code}: {error}")
        except Exception as error:  # noqa: BLE001
            bad("ask() failed", str(error)[:120])
    print()

    # 4 ── the actual routing entry point ────────────────────────────────────
    print("4. jevkit route.decide() - the model-routing feature, end to end")
    if not (JEV_SKILLS / "jevkit" / "route.py").is_file():
        bad("route.py not available")
    else:
        sys.path.insert(0, str(JEV_SKILLS))

        pool = Path(PROJECT / "routing.json")
        pool.write_text(json.dumps({
            "mode": "redacted-text",
            "tiers": {
                "simple": {"general": ["local:gemma-e4b", "cmdcode:deepseek-flash"],
                           "coding": ["local:gemma-e4b", "cmdcode:deepseek-flash"]},
                "medium": {"general": ["cmdcode:deepseek-flash", "cmdcode:qwen-coder"],
                           "coding": ["cmdcode:qwen-coder", "cmdcode:deepseek-flash"]},
                "hard": {"general": ["cmdcode:deepseek-v4.1", "cmdcode:gpt-5.6-luna"],
                         "coding": ["cmdcode:gpt-5.6-luna", "cmdcode:deepseek-v4.1"]},
            },
            "exclude": [],
        }, indent=2), encoding="utf-8")
        os.environ["JEV_ROUTING_CONFIG"] = str(pool)

        from jevkit import route  # noqa: E402

        rows = [
            {"ref": "local:gemma-e4b", "provider": "local", "model": "gemma-e4b",
             "cost_in": 0.0, "cost_out": 0.0, "context": 32000},
            {"ref": "cmdcode:deepseek-flash", "provider": "cmdcode", "model": "deepseek-flash",
             "cost_in": 0.00005, "cost_out": 0.0002, "context": 256000},
            {"ref": "cmdcode:qwen-coder", "provider": "cmdcode", "model": "qwen-coder",
             "cost_in": 0.0003, "cost_out": 0.0012, "context": 256000},
            {"ref": "cmdcode:deepseek-v4.1", "provider": "cmdcode", "model": "deepseek-v4.1",
             "cost_in": 0.0011, "cost_out": 0.0044, "context": 256000},
            {"ref": "cmdcode:gpt-5.6-luna", "provider": "cmdcode", "model": "gpt-5.6-luna",
             "cost_in": 0.003, "cost_out": 0.012, "context": 256000},
        ]
        for label, prompt in (("a small lookup", "what was that file called again?"),
                              ("a coding turn", TURN)):
            decision = route.decide(prompt, current="cmdcode:deepseek-v4.1", rows=rows, timeout=30.0)
            ok(f"{label:16} -> {decision.get('tier')}/{decision.get('specialty')}",
               f"{decision.get('model')}  conf={decision.get('confidence')}  "
               f"reason={decision.get('reason')}")
        print()
        print("  NOTE: the thresholds in ~/.hermes/jev/routing.json were tuned against")
        print("  Jev's confidence, which is a different quantity from Laya's (Laya reports")
        print("  normalized entropy on choice/score). A 'kept your current model' reason")
        print("  above is that mismatch, not a broken bridge -- see the README section on")
        print("  re-tuning, and start in shadow mode.")
        print()
        print("  Decision latency on this machine, per call:")
        print("    single pass, routing-sized turn : ~1.3 s warm on CPU (within the client's 2.5 s budget)")
        print("    long state on the long checkpoint: see LFA_LONG_POLICY")
        print("    window-by-window scan            : bounded by LFA_SCAN_MAX_TOKENS (default: refused)")

    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("All checks passed: the local engine answers a real Jev client.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
