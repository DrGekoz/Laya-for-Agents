"""The ``laya-for-agents`` command line.

    laya-for-agents serve            run the Jev-protocol server
    laya-for-agents doctor           preflight: torch, device, checkpoints, port, budget
    laya-for-agents smoke            load a checkpoint and answer one real question set
    laya-for-agents config           print the environment a client needs
    laya-for-agents setup-hermes     install hermes-jev-skills and point it here
    laya-for-agents install-gateway-hook    start this server with every gateway boot

``serve`` is the only subcommand that needs the server extras; the rest run on a
bare Python so a broken install can still be diagnosed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__ as LFA_VERSION
from .gateway_hook import (
    DISABLE_ENV as DISABLE_HOOK_ENV,
    find_hermes_home,
    gateway_hook_status,
    hermes_homes,
    install_gateway_hook,
    uninstall_gateway_hook,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR = PROJECT_ROOT / "vendor"
JEV_SKILLS_REPO = "https://github.com/kerpopule/hermes-jev-skills"
JEV_SKILLS_DIR = VENDOR / "hermes-jev-skills"


# ── helpers ──────────────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> None:
    print(f"  [ OK ] {label}" + (f"  {detail}" if detail else ""))


def _warn(label: str, detail: str = "") -> None:
    print(f"  [WARN] {label}" + (f"  {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f"  {detail}" if detail else ""))


def _hermes_homes() -> List[Path]:
    """Every Hermes home on this machine, most specific first (see gateway_hook)."""
    return hermes_homes()


def _find_hermes_home() -> Optional[Path]:
    return find_hermes_home()


def _env_value(path: Path, key: str) -> Optional[str]:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _write_env(path: Path, updates: Dict[str, str], *, dry_run: bool = False) -> List[str]:
    """Set keys in a dotenv file, preserving everything else. Returns what changed.

    Written as a read-modify-write with a backup, because this file holds live
    credentials for other tools and an append-or-clobber would be the wrong
    shape for it.
    """
    existing = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    lines = existing.splitlines()
    changed: List[str] = []
    for key, value in updates.items():
        replaced = False
        for index, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                if lines[index] == f"{key}={value}":
                    replaced = True
                    break
                lines[index] = f"{key}={value}"
                changed.append(f"{key} (updated)")
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
            changed.append(f"{key} (added)")
    if not changed or dry_run:
        return changed
    body = "\n".join(lines).rstrip("\n") + "\n"
    if path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak-laya-for-agents"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return changed


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def _base_url(host: str, port: int) -> str:
    # Plaintext is permitted by the Jev client only on numeric loopback, so the
    # advertised base URL always uses 127.0.0.1 rather than a hostname.
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "localhost") else host
    return f"http://{shown}:{port}"


def _probe_laya(host: str, port: int, timeout: float = 2.5) -> Optional[Dict[str, Any]]:
    """This server's /health body when it owns the port, else None.

    Answers "is the thing on this port *us*", which is the question both the
    doctor and the gateway hook need. A foreign listener, a proxy error page, or
    a server that answers a different protocol all read as None.
    """
    import urllib.request

    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "localhost") else host
    try:
        with urllib.request.urlopen(f"http://{shown}:{port}/health", timeout=timeout) as response:
            if response.status != 200:
                return None
            body = json.loads(response.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(body, dict) and body.get("service") == "laya-for-agents":
        return body
    return None


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        _fail("uvicorn is missing", 'install with:  pip install -e ".[serve]"')
        return 2

    from .server import create_app
    from .settings import load as load_settings

    settings = load_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"Laya for Agents  ->  {_base_url(host, port)}/v1/systemone")
    print(f"  device={settings.device or 'auto'}  single_max={settings.single_max}  "
          f"head_max_len<={settings.head_max_len_ceiling}  auto_long={settings.auto_long}")
    print("  set TYPESAFE_BASE_URL to that URL to point a Jev client at it "
          "(no API key needed)")
    uvicorn.run(create_app(settings), host=host, port=port,
                log_level=args.log_level or settings.log_level, access_log=bool(args.access_log))
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    from .settings import load as load_settings

    problems = 0
    settings = load_settings()
    print("Laya for Agents - doctor")
    print(f"project root  {PROJECT_ROOT}")
    print(f"python        {sys.version.split()[0]}  ({sys.executable})")

    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        _ok("python >= 3.10")
    else:
        problems += 1
        _fail("python >= 3.10 required", f"this is {major}.{minor}")

    for module, why in (("laya", "the decision engine"), ("torch", "checkpoint runtime"),
                        ("transformers", "checkpoint runtime"), ("fastapi", "the HTTP server"),
                        ("uvicorn", "the HTTP server")):
        try:
            found = __import__(module)
            version = getattr(found, "__version__", "?")
            _ok(f"{module} importable", f"{version}  ({why})")
        except Exception as error:  # noqa: BLE001
            level = _warn if module in ("fastapi", "uvicorn") else _fail
            level(f"{module} not importable", str(error)[:70])
            if module not in ("fastapi", "uvicorn"):
                problems += 1

    try:
        import torch

        if torch.cuda.is_available():
            _ok("CUDA available", f"{torch.cuda.get_device_name(0)}")
        else:
            _warn("CUDA not available", "checkpoints will run on CPU (correct, slower)")
    except Exception:  # noqa: BLE001
        pass

    free = _port_free(settings.host, settings.port)
    if free:
        _ok(f"port {settings.port} free")
    else:
        # The port being taken is only a problem when it is somebody else's. With
        # the gateway hook installed this server is normally already up, and a
        # doctor run that cried "problem" every time would train people to
        # ignore it.
        live = _probe_laya(settings.host, settings.port)
        if live is not None:
            state = "ready" if live.get("ready") else "loading checkpoints"
            _ok(f"port {settings.port} already serving", f"this server is up ({state})")
        else:
            problems += 1
            _fail(f"port {settings.port} already in use",
                  "another server is running, or change LFA_PORT")

    _ok("budget policy",
        f"single_max={settings.single_max}  max_token_budget={settings.max_token_budget}  "
        f"auto_long={settings.auto_long}")

    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        have = sorted({r.repo_id for r in cache.repos if "convaiinnovations" in r.repo_id})
        if have:
            _ok("checkpoints cached", ", ".join(have))
        else:
            _warn("no Laya checkpoints cached yet",
                  "the first `serve` or `smoke` downloads them (~1.7 GB)")
    except Exception:  # noqa: BLE001
        _warn("could not inspect the Hugging Face cache", "not fatal")

    home = _find_hermes_home()
    if home:
        current = _env_value(home / ".env", "TYPESAFE_BASE_URL")
        if current:
            _ok("Hermes pointed here", f"TYPESAFE_BASE_URL={current}   ({home / '.env'})")
        else:
            _warn("TYPESAFE_BASE_URL not set in the Hermes .env",
                  "run:  laya-for-agents setup-hermes")
        hook = gateway_hook_status(home)
        if hook.get("installed"):
            _ok("gateway startup hook installed",
                f"{hook['directory']}  (event {', '.join(hook.get('events', []))})")
        else:
            _warn("gateway startup hook not installed",
                  "the server only runs while you leave it running;  "
                  "run:  laya-for-agents install-gateway-hook")
    else:
        _warn("no Hermes home found", "nothing to wire up")

    print()
    if problems:
        print(f"{problems} problem(s) to fix.")
        return 1
    print("No blocking problems.")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from .engine import Engine
    from .settings import load as load_settings

    settings = load_settings()
    engine = Engine(settings)
    print("loading a checkpoint (first run downloads it)...")
    described = engine.warm()
    print(f"  resident: {', '.join(described['loaded']) or 'none'}   device: {described['device']}")
    print()

    state = args.state or (
        "Refactor the parallax renderer to stream frames straight into NVENC instead of "
        "writing a PNG per frame, and keep the output bit-identical."
    )
    questions = {
        "difficulty": {"type": "score", "instructions": "How demanding is it to complete this turn well?",
                       "criteria": ["Trivial or mechanical: a lookup, reformat, rename, short factual reply",
                                    "Routine: ordinary multi-step work with a clear path",
                                    "Substantial: needs planning, several interacting parts, debugging",
                                    "Expert: subtle, ambiguous, or high-stakes"]},
        "kind": {"type": "choice", "instructions": "What kind of work is this turn mainly?",
                 "criteria": {"coding": "writing, changing or debugging software",
                              "writing": "prose, marketing, documents",
                              "research": "finding, comparing or synthesizing information",
                              "general": "conversation, planning, operations, none of the others"}},
        "costly_mistake": {"type": "noul",
                           "instructions": "A wrong or sloppy answer here would be costly or hard to undo"},
    }
    from . import protocol

    checked = protocol.check_request({"state": state, "questions": questions}, settings)
    plan = engine.plan(checked)
    print(f"plan: mode={plan['mode']}  estimated_state_tokens={plan['estimated_state_tokens']}  "
          f"needed={plan['needed_tokens']}  single_max={plan['single_max']}")
    reply = engine.answer(checked)
    print()
    print(json.dumps({"answers": reply["answers"], "usage": reply["usage"],
                      "routing": reply.get("routing"), "laya": reply.get("laya")}, indent=2))
    print()
    print(f"checkpoint that answered: {(reply.get('routing') or {}).get('model')}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from .settings import load as load_settings

    settings = load_settings()
    base = _base_url(settings.host, settings.port)
    print("# Point a Jev client at the local engine. No API key is needed: the client")
    print("# deliberately never forwards a provider credential to an override host.")
    print(f"TYPESAFE_BASE_URL={base}")
    print("# Only if this server runs with LFA_API_KEY set:")
    print("# JEV_PROXY_API_KEY=<same value>")
    print()
    print("# Hermes plugin switches (add to each profile's config.yaml, or use /jev):")
    print("#   plugins.entries.hermes-jev.routing: shadow   # decide + log, switch nothing")
    print("#   plugins.entries.hermes-jev.skills:  on")
    print()
    print(f"# localhost is REJECTED by the client; use {base} with the numeric loopback IP.")
    return 0


def _hook_check(handler: Path) -> List[str]:
    """Import the freshly written handler and return anything wrong with it.

    Importing (rather than eyeballing) is the check that matters: the file is
    executed by the gateway with no Laya imports available, so if it cannot be
    loaded on a bare interpreter it would fail there too.
    """
    import importlib.util

    problems: List[str] = []
    spec = importlib.util.spec_from_file_location("lfa_gateway_hook_check", handler)
    if spec is None or spec.loader is None:
        return [f"cannot load {handler}"]
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        return [f"handler.py does not import: {error}"]
    for attribute in ("handle", "_target", "_health"):
        if not hasattr(module, attribute):
            problems.append(f"handler.py has no {attribute}()")
    if hasattr(module, "handle") and not _is_async(module.handle):
        problems.append("handle() is not async (the gateway awaits it)")
    if hasattr(module, "PYTHON") and not Path(module.PYTHON).exists():
        problems.append(f"baked interpreter is missing: {module.PYTHON}")
    return problems


def _is_async(function: Any) -> bool:
    import inspect

    return inspect.iscoroutinefunction(function)


def _report_hook(result: Any, *, verb: str) -> None:
    label = "would write" if result.dry_run else verb
    for name in result.changed:
        _ok(f"{label} {name}", str(result.directory / name))
    if not result.changed:
        _ok("already up to date", str(result.directory))
    if result.changed and not result.dry_run:
        problems = _hook_check(result.handler)
        if problems:
            for problem in problems:
                _fail(problem)
        else:
            _ok("handler.py loads", "handle() present (stdlib-only, gateway-safe)")


def cmd_install_gateway_hook(args: argparse.Namespace) -> int:
    """Write the gateway:startup hook so the server runs whenever Hermes does."""
    from .settings import load as load_settings

    print("Laya for Agents - install-gateway-hook")
    home: Optional[Path] = Path(args.home) if args.home else _find_hermes_home()
    if home is None:
        _fail("no Hermes home found", "pass --home <dir> if it lives somewhere unusual")
        return 2
    settings = load_settings()
    base = args.base_url or _base_url(settings.host, settings.port)
    print(f"hermes home   {home}")
    print(f"endpoint      {base}")
    try:
        result = install_gateway_hook(
            home,
            project_root=Path(args.project_root) if args.project_root else PROJECT_ROOT,
            python_exe=Path(args.python) if args.python else Path(sys.executable),
            base_url=base,
            dry_run=args.dry_run,
        )
    except RuntimeError as error:
        _fail(str(error))
        return 2

    _report_hook(result, verb="wrote")
    print(f"  [ OK ] log file  {result.log_file}")
    print()
    print("The gateway loads hooks once, at startup, from <hermes home>/hooks/.")
    print("Restart it so this takes effect:   hermes gateway restart")
    print("Fired once per boot: already-running server is left alone, a port held by")
    print(f"something else is reported and never fought over. Disable with {DISABLE_HOOK_ENV}=off.")
    return 0


def cmd_uninstall_gateway_hook(args: argparse.Namespace) -> int:
    """Remove the gateway:startup hook (the running server is left running)."""
    print("Laya for Agents - uninstall-gateway-hook")
    home: Optional[Path] = Path(args.home) if args.home else _find_hermes_home()
    if home is None:
        _fail("no Hermes home found", "pass --home <dir> if it lives somewhere unusual")
        return 2
    result = uninstall_gateway_hook(home, dry_run=args.dry_run)
    if result.changed:
        _ok("would remove" if args.dry_run else "removed", str(result.directory))
        print("  a server started by the hook keeps running until you stop it (Ctrl+C in")
        print("  its window, or close the LayaForAgents.bat console).")
    else:
        _ok("nothing to remove", "the hook was not installed")
    return 0


def cmd_setup_hermes(args: argparse.Namespace) -> int:
    """Install hermes-jev-skills and point it at this server."""
    print("Laya for Agents - setup-hermes")
    home = Path(args.home) if getattr(args, "home", None) else _find_hermes_home()
    if home is None:
        _fail("no Hermes home found", "is Hermes installed on this machine?")
        return 2
    print(f"hermes home   {home}")

    # 1. the Jev skill pack
    skills = Path(args.jev_skills) if args.jev_skills else JEV_SKILLS_DIR
    if not (skills / "install.py").is_file():
        if args.dry_run:
            print(f"  would clone {JEV_SKILLS_REPO} -> {skills}")
        else:
            VENDOR.mkdir(parents=True, exist_ok=True)
            print(f"  cloning {JEV_SKILLS_REPO} -> {skills}")
            result = subprocess.run(["git", "clone", "--depth", "1", JEV_SKILLS_REPO, str(skills)],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                _fail("clone failed", result.stderr.strip()[:200])
                return 1
    else:
        _ok("hermes-jev-skills present", str(skills))

    # 2. its installer (installs plugins + skills for every agent it finds)
    if args.dry_run:
        print(f"  would run: {sys.executable} {skills / 'install.py'} --check")
    else:
        print(f"  running {skills / 'install.py'}")
        result = subprocess.run([sys.executable, str(skills / "install.py")],
                                capture_output=True, text=True)
        tail = (result.stdout or "").strip().splitlines()[-14:]
        for line in tail:
            print(f"    {line}")
        if result.returncode != 0:
            _fail("installer failed", (result.stderr or "").strip()[:300])
            return 1

    # 3. point the client at the local engine
    from .settings import load as load_settings

    settings = load_settings()
    base = _base_url(settings.host, settings.port)
    updates = {"TYPESAFE_BASE_URL": base}
    if args.proxy_key:
        updates["JEV_PROXY_API_KEY"] = args.proxy_key
    env_path = home / ".env"
    changed = _write_env(env_path, updates, dry_run=args.dry_run)
    if changed:
        label = "would write" if args.dry_run else "wrote"
        _ok(f"{label} {env_path}", ", ".join(changed))
    else:
        _ok("Hermes .env already correct", f"TYPESAFE_BASE_URL={base}")

    # 4. keep the engine running: a gateway:startup hook, so "gateway is up"
    #    always implies "the decision server is up". Without it the plugin's
    #    calls fail open and every turn costs a timeout.
    if args.no_gateway_hook:
        _warn("gateway startup hook skipped", "--no-gateway-hook")
    else:
        try:
            hook = install_gateway_hook(
                home,
                project_root=PROJECT_ROOT,
                python_exe=Path(sys.executable),
                base_url=base,
                dry_run=args.dry_run,
            )
        except RuntimeError as error:
            _fail("gateway hook failed", str(error))
            return 1
        _report_hook(hook, verb="wrote")
        print(f"  [ OK ] log file  {hook.log_file}")

    print()
    print("Next:")
    print(f"  1. start the engine:   laya-for-agents serve")
    print(f"  2. restart the Hermes gateway so it picks up {env_path.name} and the hook")
    print( "  3. start in shadow mode:  /jev routing shadow")
    return 0

# ── entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-for-agents",
        description="Run the TypeSafe Jev /v1/systemone wire protocol on a local Laya checkpoint.")
    parser.add_argument("--version", action="version", version=f"laya-for-agents {LFA_VERSION}")
    subs = parser.add_subparsers(dest="command")

    serve = subs.add_parser("serve", help="run the decision server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--log-level", default=None)
    serve.add_argument("--access-log", action="store_true", help="log every request line")
    serve.set_defaults(func=cmd_serve)

    doctor = subs.add_parser("doctor", help="preflight this install")
    doctor.set_defaults(func=cmd_doctor)

    smoke = subs.add_parser("smoke", help="load a checkpoint and answer one real question set")
    smoke.add_argument("--state", default=None, help="the text to decide on")
    smoke.set_defaults(func=cmd_smoke)

    config = subs.add_parser("config", help="print the environment a client needs")
    config.set_defaults(func=cmd_config)

    setup = subs.add_parser("setup-hermes", help="install hermes-jev-skills, point it here, and start with the gateway")
    setup.add_argument("--home", default=None, help="Hermes home (defaults to the one the gateway uses)")
    setup.add_argument("--jev-skills", default=None, help="path to an existing hermes-jev-skills checkout")
    setup.add_argument("--proxy-key", default=None, help="value for JEV_PROXY_API_KEY (only if LFA_API_KEY is set)")
    setup.add_argument("--no-gateway-hook", action="store_true",
                       help="skip the gateway:startup hook that keeps the server running")
    setup.add_argument("--dry-run", action="store_true")
    setup.set_defaults(func=cmd_setup_hermes)

    hook = subs.add_parser("install-gateway-hook",
                           help="install the gateway:startup hook that runs this server with Hermes")
    hook.add_argument("--home", default=None, help="Hermes home (defaults to the one the gateway uses)")
    hook.add_argument("--project-root", default=None, help="checkout to serve from (defaults to this one)")
    hook.add_argument("--python", default=None, help="interpreter for the server (defaults to this one)")
    hook.add_argument("--base-url", default=None, help="endpoint to serve (defaults to this install's)")
    hook.add_argument("--dry-run", action="store_true")
    hook.set_defaults(func=cmd_install_gateway_hook)

    unhook = subs.add_parser("uninstall-gateway-hook", help="remove that hook again")
    unhook.add_argument("--home", default=None, help="Hermes home (defaults to the one the gateway uses)")
    unhook.add_argument("--dry-run", action="store_true")
    unhook.set_defaults(func=cmd_uninstall_gateway_hook)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
