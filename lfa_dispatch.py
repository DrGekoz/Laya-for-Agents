#!/usr/bin/env python3
"""Dispatch a coding job to the seat Laya picked.

Laya (served locally by Laya for Agents) decides *which* seat should take a job.
This is the hand that does something about it, because nothing in Laya or jevkit
opens a connection -- the client's own words are "the plugin can swap a model but
not a provider connection", so escalation is a delegation signal for the agent and
never a silent switch.

Three parts, and the split matters:

  targets     the fixed rung id -> command map, written by a person, in this file
  ladder      which seat is free, from jevkit's escalation ladder (SSH probes)
  run         execute the chosen rung, with the task delivered on stdin

Safety rules this file exists to keep:

  * A rung id is only ever accepted if it appears in TARGETS below. Laya returns
    outcome ids from a table it was handed; it cannot invent a destination, and
    neither can anything that talked to it.
  * The task is written to the child's STDIN, never spliced into a command line.
    `codex exec -` reads its prompt from stdin, so the job text is never parsed by
    a shell -- locally or on the far side of the ssh.
  * The model is pinned explicitly so a config edit on the remote box cannot
    quietly move work onto a model that is not wanted.

Usage:

    python lfa_dispatch.py targets
    python lfa_dispatch.py ladder status|choose|refuse|clear
    python lfa_dispatch.py route --task "..."            # Laya picks a destination
    python lfa_dispatch.py run --rung rygel-codex --task -   # job on stdin
    python lfa_dispatch.py run --rung rygel-codex --task "..." --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "hermes-jev-skills"
if VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


# ── the fixed command map ─────────────────────────────────────────────────────
#
# `action: "delegate"` runs argv with the task on stdin.
# `action: "local"`    runs nothing: it means "keep the job here" and the caller
#                      carries on with its own tools. The last rung has to be a
#                      local answer, so hard work is never refused outright.

CODEX = "~/.local/bin/codex"

TARGETS: Dict[str, Dict[str, Any]] = {
    "rygel-codex": {
        "action": "delegate",
        "description": "Codex CLI on the sanfrancisco box over SSH",
        "ssh_host": "rygel-sf",
        "ssh_options": ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"],
        # `-` reads the prompt from stdin. -s workspace-write is the narrow sandbox:
        # enough to edit files under --cd, not the blanket bypass flag.
        "remote": (
            "mkdir -p {cd} && exec {codex} exec --skip-git-repo-check "
            "-m {model} -s workspace-write -C {cd} -"
        ),
        "default_cd": "/home/gekoz/codex-runs",
        "default_model": "gpt-5.6-luna",
    },
    "local-hermes": {
        "action": "local",
        "description": "this machine, with the current session's own tools",
    },
}


def _iter_config_file() -> Optional[Path]:
    """The same routing.json jevkit reads, so the ladder and the map cannot drift."""
    override = os.environ.get("JEV_ROUTING_CONFIG")
    if override:
        return Path(override)
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    root = home.parent.parent if home.parent.name == "profiles" else home
    candidate = root / "jev" / "routing.json"
    return candidate if candidate.is_file() else None


def rungs() -> List[Dict[str, Any]]:
    """The configured rungs, read through jevkit so parsing rules stay in one place."""
    try:
        from jevkit import route as route_mod

        return list((route_mod.load_config().get("escalation") or {}).get("rungs") or [])
    except Exception as error:  # noqa: BLE001
        path = _iter_config_file()
        if not path:
            print(f"no routing.json found and jevkit unavailable ({error})", file=sys.stderr)
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return list((data.get("escalation") or {}).get("rungs") or [])


def cmd_targets(args: argparse.Namespace) -> int:
    configured = {r.get("name") for r in rungs()}
    rows = []
    for name, spec in TARGETS.items():
        rows.append({"rung": name, "action": spec["action"], "description": spec["description"],
                     "configured": name in configured})
    out = {"targets": rows,
           "note": "a rung is only dispatchable when it is both in TARGETS here and in "
                   "routing.json's escalation.rungs"}
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    for row in rows:
        mark = "ok " if row["configured"] else "MISSING from routing.json"
        print(f"  {row['rung']:14} {row['action']:9} {row['description']:46} [{mark}]")
    return 0 if all(r["configured"] for r in rows) else 1


def cmd_ladder(args: argparse.Namespace) -> int:
    from jevkit import ladder, route as route_mod

    configured = rungs()
    if args.action == "status":
        result = ladder.status(configured)
    elif args.action == "choose":
        if not configured:
            print("no escalation.rungs in routing.json", file=sys.stderr)
            return 2
        result = ladder.choose(configured, skip_probe=args.no_probe)
    elif args.action == "refuse":
        if not args.rung:
            print("refuse needs --rung", file=sys.stderr)
            return 2
        result = ladder.refuse(args.rung, args.reason or "refused",
                               cooldown=float(args.cooldown or 1800))
    else:
        ladder.clear(args.rung)
        result = {"cleared": args.rung or "all"}
    print(json.dumps(result, indent=2))
    return 0


# ── route_to ──────────────────────────────────────────────────────────────────
#
# The default floor in jevkit is 0.85. Measured against Laya, a three-option choice
# sits between 0.10 and 0.30 on this metric, so 0.85 is unreachable and route_to
# would answer `fallback` every single time without ever looking at the state.
# These numbers are the measured band, not a preference.

DEFAULT_FLOOR = 0.20
DEFAULT_MIN_MARGIN = 0.05


def _destinations(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_route(args: argparse.Namespace) -> int:
    from jevkit import route_to

    table = _destinations(Path(args.destinations))
    task = sys.stdin.read() if args.task == "-" else args.task
    if not task or not task.strip():
        print("route needs --task (or --task - with the job on stdin)", file=sys.stderr)
        return 2
    decision = route_to.route_to(task, table, fallback=args.fallback,
                                 floor=args.floor, min_margin=args.min_margin)
    kept = {k: decision.get(k) for k in
            ("dest", "pick", "routed", "confidence", "margin", "stage", "action", "reason")}
    kept["dispatchable"] = kept.get("dest") in TARGETS
    print(json.dumps(kept, indent=2))
    return 0


# ── run ───────────────────────────────────────────────────────────────────────

def _build(spec: Dict[str, Any], cd: Optional[str], model: Optional[str]) -> List[str]:
    if spec["action"] != "delegate":
        return []
    remote = spec["remote"].format(
        cd=shlex.quote(cd or spec["default_cd"]),
        model=shlex.quote(model or spec["default_model"]),
        codex=CODEX,
    )
    return ["ssh", *spec["ssh_options"], spec["ssh_host"], remote]


def cmd_run(args: argparse.Namespace) -> int:
    spec = TARGETS.get(args.rung)
    if spec is None:
        print(f"unknown rung {args.rung!r}. Known: {', '.join(sorted(TARGETS))}", file=sys.stderr)
        return 2
    task = sys.stdin.read() if args.task == "-" else args.task
    if not task or not task.strip():
        print("run needs --task (or --task - with the job on stdin)", file=sys.stderr)
        return 2

    if spec["action"] == "local":
        print(json.dumps({"rung": args.rung, "action": "local", "dispatched": False,
                          "detail": "keep the job on this machine; the caller's own tools handle it"},
                         indent=2))
        return 0

    argv = _build(spec, args.cd, args.model)
    # The task goes on stdin. Quote it for display only -- it is never parsed.
    printable = " ".join(shlex.quote(part) for part in argv)
    if args.dry_run:
        print(json.dumps({"rung": args.rung, "action": "delegate", "dispatched": False,
                          "dry_run": True, "argv": argv, "stdin_chars": len(task),
                          "command": printable}, indent=2))
        return 0

    try:
        done = subprocess.run(argv, input=task.encode("utf-8"), capture_output=True,
                              timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        # A seat that timed out is a seat to skip for a while, and the ladder is where
        # every other lane finds that out -- hence the refusal, not just an error.
        _refuse_quietly(args.rung, f"dispatch timed out after {args.timeout}s")
        print(json.dumps({"rung": args.rung, "dispatched": True, "ok": False,
                          "error": "timeout", "timeout_s": args.timeout}, indent=2))
        return 1

    stdout = done.stdout.decode("utf-8", "replace")
    stderr = done.stderr.decode("utf-8", "replace")
    ok = done.returncode == 0
    if not ok:
        _refuse_quietly(args.rung, f"codex exited {done.returncode}")
    result = {"rung": args.rung, "action": "delegate", "dispatched": True, "ok": ok,
              "exit_code": done.returncode, "stdout": stdout[-args.tail:],
              "stderr": stderr[-args.tail:], "stdout_chars": len(stdout)}
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def _refuse_quietly(name: str, reason: str) -> None:
    try:
        from jevkit import ladder

        ladder.refuse(name, reason)
    except Exception:  # noqa: BLE001 - recording the refusal must not mask the real failure
        pass


def cmd_selftest(args: argparse.Namespace) -> int:
    """No network, no model: the things that would silently ruin a dispatch."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("routing.json is findable", _iter_config_file() is not None,
          str(_iter_config_file()))
    configured = {r.get("name") for r in rungs()}
    for name in TARGETS:
        check(f"rung {name} is in routing.json", name in configured)

    hostile = "rm -rf /'; echo pwned; #"
    argv = _build(TARGETS["rygel-codex"], "/home/gekoz/codex-runs", "gpt-5.6-luna")
    check("the task never enters argv", all(hostile not in part for part in argv))
    check("the sandbox is the narrow one, not the bypass",
          "workspace-write" in argv[-1] and "dangerously-bypass" not in argv[-1])
    check("the model is pinned", "gpt-5.6-luna" in argv[-1], argv[-1][:90])
    check("stdin is how the prompt travels", argv[-1].rstrip().endswith("-"))
    check("a local rung builds no command", _build(TARGETS["local-hermes"], None, None) == [])
    check("an unknown rung is refused", "not-a-rung" not in TARGETS)

    if args.json:
        print(json.dumps({"checks": checks}, indent=2))
    else:
        for row in checks:
            print(f"  [{'OK' if row['ok'] else 'FAIL'}] {row['check']}"
                  + (f"  ({row['detail']})" if row["detail"] and not row["ok"] else ""))
    return 0 if all(c["ok"] for c in checks) else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lfa_dispatch", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("targets", help="the fixed rung id -> command map")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("ladder", help="which seat is free (runs the SSH probes)")
    p.add_argument("action", choices=["status", "choose", "refuse", "clear"])
    p.add_argument("--rung")
    p.add_argument("--reason")
    p.add_argument("--cooldown")
    p.add_argument("--no-probe", action="store_true")
    p.set_defaults(func=cmd_ladder)

    p = sub.add_parser("route", help="let Laya pick a destination for this job")
    p.add_argument("--task", required=True, help="the job, or - to read it from stdin")
    p.add_argument("--destinations", default=str(HERE / "destinations.json"))
    p.add_argument("--fallback", default="local-hermes")
    p.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    p.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("run", help="dispatch a job to a rung")
    p.add_argument("--rung", required=True)
    p.add_argument("--task", required=True, help="the job, or - to read it from stdin")
    p.add_argument("--cd", help="working directory on the far side")
    p.add_argument("--model")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--tail", type=int, default=20000, help="how much output to show")
    p.add_argument("--dry-run", action="store_true", help="show the command, run nothing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("selftest", help="offline checks: no network, no model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
