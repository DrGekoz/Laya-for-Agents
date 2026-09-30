"""HTTP-surface tests: the real ASGI app over a real socket, with a fake engine.

These run the actual FastAPI app through uvicorn on an ephemeral port and talk to
it with urllib, so the middleware, the auth gate, the status codes and the JSON
envelopes are exercised exactly as a client would. A fake engine stands in for a
checkpoint, so no torch and no download is involved.

Skipped (not failed) when the server extras are absent, so the offline suite still
runs on a bare Python.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import uvicorn  # noqa: F401

    HAVE_SERVER = True
except Exception:  # noqa: BLE001
    HAVE_SERVER = False

from laya_for_agents.settings import Settings  # noqa: E402

if HAVE_SERVER:
    from laya_for_agents.server import create_app
else:  # keep the module importable on a bare Python
    create_app = None  # type: ignore[assignment]

class FakeEngine:
    """Minimal stand-in for Engine: enough for the HTTP layer to be real."""

    def __init__(self):
        self.settings = Settings()
        self.seen = []

    def warm(self):
        return self.describe()

    def describe(self):
        return {"loaded": ["english"], "loaded_revisions": {"english": "test"}, "device": "cpu"}

    def health(self):
        return {"status": "ok", "service": "laya-for-agents", "loaded": ["english"],
                "device": "cpu", "protocol": "Jev /v1/systemone",
                "budget": {"single_max": self.settings.single_max}, "stats": {}, "uptime_s": 0}

    def plan(self, request):
        return {"mode": "single"}

    def answer(self, request):
        self.seen.append(request)
        return {
            "model": "laya-for-agents",
            "answers": {"q": {"type": "noul", "noul": 0.5, "confidence": 0.5}},
            "usage": {"input_tokens": 3, "output_tokens": 0},
            "routing": {"model": "english"},
            "laya": {"mode": "single"},
        }


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(HAVE_SERVER, 'needs the server extras: pip install -e ".[serve]"')
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from laya_for_agents.server import create_app

        cls.port = _free_port()
        cls.engine = FakeEngine()
        config = uvicorn.Config(create_app(Settings(host="127.0.0.1", port=cls.port), cls.engine),
                                host="127.0.0.1", port=cls.port, log_level="error")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                cls._get("/health")
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.15)
        raise RuntimeError("server did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    # ── helpers ──────────────────────────────────────────────────────────────

    @classmethod
    def _url(cls, path):
        return f"http://127.0.0.1:{cls.port}{path}"

    @classmethod
    def _get(cls, path):
        with urllib.request.urlopen(cls._url(path), timeout=10) as reply:
            return reply.status, json.loads(reply.read())

    @classmethod
    def _post(cls, path, payload, headers=None):
        body = json.dumps(payload).encode()
        request = urllib.request.Request(cls._url(path), data=body, method="POST",
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=20) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                return error.code, json.loads(raw)
            except Exception:  # noqa: BLE001
                return error.code, {"raw": raw.decode(errors="replace")}

    # ── tests ────────────────────────────────────────────────────────────────

    def test_health_is_open_and_reports_the_protocol(self):
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["protocol"], "Jev /v1/systemone")

    def test_index_describes_the_service(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertEqual(body["endpoints"]["decide"], "POST /v1/systemone")

    def test_a_valid_request_is_answered(self):
        status, body = self._post("/v1/systemone", {
            "state": "hello", "questions": {"q": {"type": "noul", "instructions": "risky?"}}})
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["q"]["noul"], 0.5)
        for key in ("model", "answers", "usage", "routing"):
            self.assertIn(key, body)

    def test_the_body_reaches_the_engine_validated(self):
        self.engine.seen.clear()
        self._post("/v1/systemone", {
            "state": "hello", "questions": {"q": {"type": "noul", "instructions": "risky?"}}})
        self.assertEqual(self.engine.seen[-1]["state"], "hello")
        self.assertIn("_encoded", self.engine.seen[-1])

    def test_a_missing_state_is_a_400(self):
        status, body = self._post("/v1/systemone",
                                  {"questions": {"q": {"type": "noul", "instructions": "x"}}})
        self.assertEqual(status, 400)
        self.assertIn("state", body["detail"])

    def test_an_oversize_choice_is_a_413(self):
        criteria = {f"o{i}": "d" for i in range(120)}
        status, _ = self._post("/v1/systemone", {
            "state": "hello",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": criteria}}})
        self.assertEqual(status, 413)

    def test_a_refused_hook_is_a_422(self):
        status, body = self._post("/v1/systemone", {
            "state": "hello", "hooks": ["x"],
            "questions": {"q": {"type": "noul", "instructions": "x"}}})
        self.assertEqual(status, 422)
        self.assertIn("cannot be sent here", body["detail"])

    def test_invalid_json_is_a_400(self):
        request = urllib.request.Request(self._url("/v1/systemone"), data=b"{not json",
                                         method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as reply:
                status = reply.status
        except urllib.error.HTTPError as error:
            status = error.code
        self.assertEqual(status, 400)

    def test_a_protocol_error_from_the_engine_keeps_its_status(self):
        from laya_for_agents import protocol

        class Angry(FakeEngine):
            def answer(self, request):
                raise protocol.ProtocolError(422, "question 'q' is too wide")

        self.engine.answer_original = self.engine.answer
        self.engine.answer = Angry().answer
        try:
            status, body = self._post("/v1/systemone", {
                "state": "hello", "questions": {"q": {"type": "noul", "instructions": "x"}}})
            self.assertEqual(status, 422)
            self.assertIn("too wide", body["detail"])
        finally:
            self.engine.answer = self.engine.answer_original

    def test_the_unknown_route_is_a_404(self):
        try:
            self._get("/v1/nothing")
            self.fail("expected a 404")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 404)


class SlowLoadTests(unittest.TestCase):
    """The defect this pins: if the preload is awaited in the ASGI lifespan, uvicorn
    never binds the port until every checkpoint is built -- 92 s warm, 218 s cold.
    For that whole window the service is unreachable, so a watchdog kills it in a
    loop and a cron health check reports it down. /health has to answer while the
    weights are still streaming in.
    """

    def test_health_answers_while_a_load_is_in_flight(self):
        release = threading.Event()

        class SlowEngine(FakeEngine):
            def warm(self):
                release.wait(timeout=30)      # stands in for the checkpoint load
                return self.describe()

            def health(self):
                body = super().health()
                body["loading"] = not release.is_set()
                body["ready"] = release.is_set()
                return body

        engine = SlowEngine()
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(Settings(host="127.0.0.1", port=port), engine),
                                              host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            # The load is still blocked. /health must still answer.
            deadline = time.time() + 20
            body = None
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as reply:
                        self.assertEqual(reply.status, 200)
                        body = json.loads(reply.read())
                        break
                except Exception:  # noqa: BLE001
                    time.sleep(0.2)
            self.assertIsNotNone(body, "the server did not answer /health during the load")
            self.assertTrue(body["loading"])
            self.assertFalse(body["ready"])
            self.assertEqual(body["status"], "ok")

            # Release the load and confirm it flips to ready.
            release.set()
            deadline = time.time() + 20
            while time.time() < deadline:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as reply:
                    body = json.loads(reply.read())
                if body["ready"]:
                    break
                time.sleep(0.2)
            self.assertTrue(body["ready"], "ready never became true after the load finished")
            self.assertFalse(body["loading"])
        finally:
            release.set()
            server.should_exit = True
            thread.join(timeout=10)


@unittest.skipUnless(HAVE_SERVER, 'needs the server extras: pip install -e ".[serve]"')
class AuthTests(unittest.TestCase):
    """A separate app, because the key is read at construction time."""

    @classmethod
    def setUpClass(cls):
        from laya_for_agents.server import create_app

        cls.port = _free_port()
        cls.engine = FakeEngine()
        settings = Settings(host="127.0.0.1", port=cls.port, api_key="s3cret")
        config = uvicorn.Config(create_app(settings, cls.engine), host="127.0.0.1",
                                port=cls.port, log_level="error")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/health", timeout=5).read()
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.15)
        raise RuntimeError("server did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def _post(self, headers):
        body = json.dumps({"state": "hello",
                           "questions": {"q": {"type": "noul", "instructions": "x"}}}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/systemone", data=body,
                                         method="POST",
                                         headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(request, timeout=10) as reply:
                return reply.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_no_bearer_is_a_401(self):
        self.assertEqual(self._post({}), 401)

    def test_a_wrong_bearer_is_a_401(self):
        self.assertEqual(self._post({"Authorization": "Bearer nope"}), 401)

    def test_the_right_bearer_is_accepted(self):
        self.assertEqual(self._post({"Authorization": "Bearer s3cret"}), 200)

    def test_health_stays_open_even_with_a_key_set(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as reply:
            self.assertEqual(reply.status, 200)


if __name__ == "__main__":
    unittest.main()
