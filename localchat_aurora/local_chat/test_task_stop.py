"""Pause/cancel must interrupt the Ollama request in flight instead of waiting for the chunk to finish."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from . import server, task_service, tasks
except ImportError:  # pragma: no cover
    import server, task_service, tasks


class SlowOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            for _ in range(200):                           # ~20 s of tokens unless the client hangs up
                self.wfile.write(b'{"message":{"content":"x"},"done":false}\n')
                self.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass

    def log_message(self, *a):
        pass


class StopTests(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), SlowOllama)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.old_url = server.OLLAMA_URL
        server.OLLAMA_URL = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        server.OLLAMA_URL = self.old_url
        self.httpd.shutdown()

    def test_streaming_call_aborts_quickly(self):
        stop = threading.Event()
        threading.Timer(0.5, stop.set).start()
        started = time.time()
        with self.assertRaises(InterruptedError):
            server._ollama_chat_cancellable({"model": "m", "messages": []}, stop)
        self.assertLess(time.time() - started, 3)

    def test_service_fallback_without_stream_function(self):
        release = threading.Event()
        service = task_service.TaskService(lambda: Path("."), lambda *a, **k: (release.wait(20), {})[1], lambda m: 4096)
        stop = threading.Event()
        threading.Timer(0.4, stop.set).start()
        started = time.time()
        with self.assertRaises(tasks.TaskCancelled):
            service._chat({"model": "m"}, stop)
        release.set()
        self.assertLess(time.time() - started, 3)

    def test_pause_mid_chunk_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            service = task_service.TaskService(lambda: Path(root), lambda *a, **k: {}, lambda m: 4096,
                                               lambda m: False, lambda m: True, server._ollama_chat_cancellable)
            progress = service.create("a.txt", ("Hello world. " * 400).encode(), {"kind": "translate", "model": "m"})
            task_id = progress["id"]
            for _ in range(150):                            # wait until the model is really streaming tokens
                live = service.manager.progress(task_id).get("live") or {}
                if live.get("phase") == "generating":
                    break
                time.sleep(0.1)
            time.sleep(0.3)
            started = time.time()
            service.action(task_id, "pause")
            self.assertEqual(service.manager.progress(task_id)["stop_requested"], "pause")
            service.manager.wait(task_id, 10)
            self.assertLess(time.time() - started, 5)
            self.assertEqual(service.manager.progress(task_id)["status"], "paused")


if __name__ == "__main__":
    unittest.main()
