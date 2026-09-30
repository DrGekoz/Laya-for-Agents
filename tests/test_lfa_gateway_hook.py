"""The Hermes gateway startup hook: what it writes, and that it would run.

Pointing Hermes at this server (``setup-hermes``) only matters if the server is
listening whenever the gateway is.  That is what the ``gateway:startup`` hook
does, so these tests cover the two ways it can be wrong: the files it writes, and
the decisions the written handler makes when it fires.

The handler is exercised by importing the *generated* file (never the template),
on a bare interpreter, which is exactly how the gateway loads it.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya_for_agents import gateway_hook  # noqa: E402
from laya_for_agents.gateway_hook import (  # noqa: E402
    HANDLER_MARKER,
    HOOK_EVENTS,
    HOOK_NAME,
    find_hermes_home,
    gateway_hook_status,
    install_gateway_hook,
    uninstall_gateway_hook,
)

# A loopback port nothing serves, so "nothing is listening" is the answer the
# handler sees no matter what the developer happens to be running on 8000.
DEAD_URL = "http://127.0.0.1:9"


def _load_handler(path: Path):
    spec = importlib.util.spec_from_file_location("lfa_hook_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _ForeignServer:
    """An HTTP server on an ephemeral port that answers /health, but not as us."""

    def __init__(self, body: str) -> None:
        import http.server
        import threading

        payload = body.encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib naming
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep the test output clean
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _foreign_health_server() -> _ForeignServer:
    return _ForeignServer('{"status": "ok", "ready": true}')


def _laya_health_server() -> _ForeignServer:
    return _ForeignServer('{"status": "ok", "service": "laya-for-agents", "ready": true}')


_ENV_KEYS = ("TYPESAFE_BASE_URL", "LFA_HOST", "LFA_PORT", "LFA_GATEWAY_AUTOSTART",
             "LFA_PROJECT_ROOT", "LFA_PYTHON")


def _isolate_env(case: unittest.TestCase) -> None:
    """Unset the variables these tests read, and put them back afterwards.

    A developer machine that already points Hermes at a *running* Laya would
    otherwise change what the handler decides, which is how a green suite turns
    into a red one on exactly the box the feature is for.
    """
    saved = {key: os.environ.pop(key) for key in _ENV_KEYS if key in os.environ}

    def restore() -> None:
        os.environ.update(saved)

    case.addCleanup(restore)


class HookInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        _isolate_env(self)
        self.tmp = Path(tempfile.mkdtemp(prefix="lfa-hook-"))
        self.home = self.tmp / "hermes-home"
        self.project = self.tmp / "checkout"
        self.project.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _install(self, **kwargs):
        params = dict(project_root=self.project, python_exe=Path(sys.executable), base_url=DEAD_URL)
        params.update(kwargs)
        return install_gateway_hook(self.home, **params)

    # ── what it writes ───────────────────────────────────────────────────────

    def test_writes_a_manifest_and_handler_the_gateway_will_load(self) -> None:
        result = self._install()

        self.assertEqual(result.directory, self.home / "hooks" / HOOK_NAME)
        self.assertTrue(result.manifest.is_file())
        self.assertTrue(result.handler.is_file())
        self.assertEqual(sorted(result.changed), ["HOOK.yaml", "handler.py"])

        manifest = result.manifest.read_text(encoding="utf-8")
        self.assertIn(f"name: {HOOK_NAME}", manifest)
        self.assertIn("- gateway:startup", manifest)
        self.assertEqual(HOOK_EVENTS, ("gateway:startup",))

    def test_handler_is_stdlib_only_and_loads_on_a_bare_interpreter(self) -> None:
        result = self._install()
        text = result.handler.read_text(encoding="utf-8")

        # the gateway process has no Laya (and no torch) importable
        for forbidden in ("import torch", "from laya", "import laya"):
            self.assertNotIn(forbidden, text)
        self.assertIn(HANDLER_MARKER, text)

        module = _load_handler(result.handler)
        self.assertTrue(asyncio.iscoroutinefunction(module.handle))
        self.assertEqual(Path(module.PYTHON), Path(sys.executable))
        self.assertEqual(Path(module.PROJECT_ROOT), self.project)
        self.assertEqual(module.BASE_URL, DEAD_URL)
        self.assertEqual(module.HOOK_VERSION, gateway_hook._version())

    def test_bakes_the_endpoint_and_follows_a_loopback_override(self) -> None:
        result = self._install()
        module = _load_handler(result.handler)

        self.assertEqual(module._target()[2], DEAD_URL)

        previous = os.environ.get("TYPESAFE_BASE_URL")
        os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:8123"
        try:
            self.assertEqual(module._target(), ("127.0.0.1", 8123, "http://127.0.0.1:8123"))
            # a non-loopback override is not this server's business: keep the baked one
            os.environ["TYPESAFE_BASE_URL"] = "https://api.typesafe.ai"
            self.assertEqual(module._target()[2], DEAD_URL)
        finally:
            if previous is None:
                os.environ.pop("TYPESAFE_BASE_URL", None)
            else:
                os.environ["TYPESAFE_BASE_URL"] = previous

    def test_reinstall_is_idempotent_and_reports_a_changed_endpoint(self) -> None:
        first = self._install()
        self.assertEqual(self._install().changed, [])

        moved = self._install(base_url="http://127.0.0.1:8123")
        self.assertEqual(moved.changed, ["handler.py"])
        self.assertIn("BASE_URL = \"http://127.0.0.1:8123\"", moved.handler.read_text(encoding="utf-8"))
        self.assertTrue(first.handler.is_file())

    def test_dry_run_writes_nothing(self) -> None:
        result = self._install(dry_run=True)
        self.assertEqual(sorted(result.changed), ["HOOK.yaml", "handler.py"])
        self.assertFalse(result.directory.exists())

    def test_uninstall_removes_the_directory_and_is_safe_twice(self) -> None:
        result = self._install()
        self.assertTrue(result.installed)

        removed = uninstall_gateway_hook(self.home)
        self.assertTrue(removed.changed)
        self.assertFalse(result.directory.exists())
        self.assertEqual(uninstall_gateway_hook(self.home).changed, [])

    def test_status_reports_the_endpoint_and_version(self) -> None:
        self._install()
        status = gateway_hook_status(self.home)

        self.assertTrue(status["installed"])
        self.assertEqual(status["base_url"], DEAD_URL)
        self.assertEqual(status["hook_version"], gateway_hook._version())
        self.assertEqual(status["events"], ["gateway:startup"])

        uninstall_gateway_hook(self.home)
        self.assertFalse(gateway_hook_status(self.home)["installed"])

    # ── what the handler does when it fires ──────────────────────────────────

    def test_handle_starts_nothing_when_the_interpreter_is_gone(self) -> None:
        result = self._install(python_exe=self.tmp / "no-such-python.exe")
        module = _load_handler(result.handler)

        asyncio.run(module.handle("gateway:startup", {}))  # must not raise

        log = result.log_file.read_text(encoding="utf-8")
        self.assertIn("interpreter missing", log)
        self.assertNotIn("spawned pid", log)

    def test_handle_can_be_turned_off_from_the_environment(self) -> None:
        result = self._install(python_exe=self.tmp / "no-such-python.exe")
        module = _load_handler(result.handler)

        os.environ["LFA_GATEWAY_AUTOSTART"] = "off"
        try:
            asyncio.run(module.handle("gateway:startup", {}))
        finally:
            os.environ.pop("LFA_GATEWAY_AUTOSTART", None)

        self.assertIn("not starting the server", result.log_file.read_text(encoding="utf-8"))

    def test_a_live_server_is_left_alone(self) -> None:
        """The already-running branch, against a real socket answering as Laya."""
        result = self._install()
        module = _load_handler(result.handler)
        live = _laya_health_server()
        self.addCleanup(live.shutdown)
        module.BASE_URL = f"http://127.0.0.1:{live.port}"
        self.assertIsNotNone(module._health(module.BASE_URL))

        asyncio.run(module.handle("gateway:startup", {}))

        log = result.log_file.read_text(encoding="utf-8")
        self.assertIn("already running", log)
        self.assertNotIn("spawned pid", log)

    def test_a_foreign_listener_on_the_port_is_never_fought_over(self) -> None:
        """A 200 that is not this server does not count as 'already running'."""
        result = self._install()
        module = _load_handler(result.handler)

        # the real _health, against a real socket that answers like somebody else
        foreign = _foreign_health_server()
        self.addCleanup(foreign.shutdown)
        module.BASE_URL = f"http://127.0.0.1:{foreign.port}"
        self.assertIsNone(module._health(module.BASE_URL))
        self.assertTrue(module._port_open("127.0.0.1", foreign.port))

        module.PYTHON = Path(self.tmp / "no-such-python.exe")
        asyncio.run(module.handle("gateway:startup", {}))

        log = result.log_file.read_text(encoding="utf-8")
        self.assertIn("busy but does not answer as Laya", log)
        self.assertNotIn("spawned pid", log)

    # ── which home it targets ────────────────────────────────────────────────

    def test_find_hermes_home_prefers_the_environment(self) -> None:
        previous = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        (self.home).mkdir(parents=True, exist_ok=True)
        (self.home / "config.yaml").write_text("plugins: {}\n", encoding="utf-8")
        try:
            self.assertEqual(find_hermes_home(), self.home)
            self.assertEqual(gateway_hook.hermes_homes()[0], self.home)
        finally:
            if previous is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous

    def test_install_without_a_home_is_refused(self) -> None:
        previous = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.tmp / "nowhere")
        real_homes = gateway_hook.hermes_homes
        gateway_hook.hermes_homes = lambda: [self.tmp / "nowhere"]
        try:
            with self.assertRaises(RuntimeError):
                install_gateway_hook(None)
        finally:
            gateway_hook.hermes_homes = real_homes
            if previous is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous


class DoctorProbeTests(unittest.TestCase):
    """Is the thing on this port *us*? The question doctor and the hook share."""

    def setUp(self) -> None:
        from laya_for_agents.cli import _probe_laya

        self.probe = _probe_laya

    def test_recognises_this_server(self) -> None:
        live = _laya_health_server()
        self.addCleanup(live.shutdown)
        body = self.probe("127.0.0.1", live.port)
        self.assertIsNotNone(body)
        self.assertTrue(body["ready"])

    def test_a_foreign_listener_is_not_this_server(self) -> None:
        foreign = _foreign_health_server()
        self.addCleanup(foreign.shutdown)
        self.assertIsNone(self.probe("127.0.0.1", foreign.port))

    def test_a_closed_port_is_not_this_server(self) -> None:
        self.assertIsNone(self.probe("127.0.0.1", 9))


class CliTests(unittest.TestCase):
    """The commands a person actually runs after `pip install -e .`."""

    def setUp(self) -> None:
        from laya_for_agents.cli import main

        _isolate_env(self)
        self.main = main
        self.tmp = Path(tempfile.mkdtemp(prefix="lfa-hook-cli-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"

    def _run(self, argv):
        """Run the CLI with its report captured, so the suite output stays readable."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self.main(argv)
        return code, buffer.getvalue()

    def test_install_then_uninstall_through_the_cli(self) -> None:
        code, output = self._run(["install-gateway-hook", "--home", str(self.home)])
        handler = self.home / "hooks" / HOOK_NAME / "handler.py"
        self.assertEqual(code, 0)
        self.assertTrue(handler.is_file())
        self.assertIn("handler.py loads", output)

        code, output = self._run(["uninstall-gateway-hook", "--home", str(self.home)])
        self.assertEqual(code, 0)
        self.assertFalse(handler.exists())
        self.assertIn("removed", output)

    def test_setup_hermes_dry_run_touches_nothing(self) -> None:
        # --dry-run must survive the parts that need the network (clone/installer)
        code, output = self._run(["setup-hermes", "--home", str(self.home), "--dry-run",
                                  "--jev-skills", str(self.tmp / "checkout")])
        self.assertEqual(code, 0)
        self.assertFalse((self.home / "hooks").exists())
        self.assertFalse((self.home / ".env").exists())
        self.assertIn("would write", output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
