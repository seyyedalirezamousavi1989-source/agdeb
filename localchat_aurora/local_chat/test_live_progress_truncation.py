"""Phase 15: a slow local model must look alive (live token progress) and a cut-off answer must be split, not accepted."""
import json
import re
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from local_chat import docengine, server as srv, tasks, task_service
from local_chat.test_translation_e2e import make_pdf, fa_token, canvas


class FakeOllama(BaseHTTPRequestHandler):
    """Streams /api/chat slowly; answers with done_reason=length when the prompt text is long (like num_predict overflow)."""
    calls = []
    long_limit = 1500          # characters of source text that still fit into "num_predict"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][-1]["content"]
        source = content.split("متن برای ترجمه:\n\n", 1)[1]
        FakeOllama.calls.append(len(source))
        text = re.sub(r"word(\d+)_(\d+)_(\d+)", lambda m: fa_token(m.group(1) + m.group(2) + m.group(3)), source)
        text = re.sub(r"row(\d+)", r"ردیف \1", text).replace("Heading of page", "عنوان صفحه")
        truncated = len(source) > FakeOllama.long_limit
        if truncated:
            text = text[: len(text) // 3]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        words = text.split(" ")
        for i, word in enumerate(words):
            line = json.dumps({"message": {"role": "assistant", "content": word + " "}, "done": False}, ensure_ascii=False)
            self.wfile.write((line + "\n").encode()); self.wfile.flush()
            if i < 30:
                time.sleep(0.01)
        end = json.dumps({"message": {"role": "assistant", "content": ""}, "done": True,
                          "done_reason": "length" if truncated else "stop"})
        self.wfile.write((end + "\n").encode()); self.wfile.flush()


@unittest.skipIf(canvas is None, "reportlab not installed")
class LiveProgressTests(unittest.TestCase):
    def setUp(self):
        FakeOllama.calls = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.patches = [patch.object(srv, "OLLAMA_URL", url), patch.object(srv, "APP_DATA_DIR", Path(self.tmp.name)),
                        patch.object(srv, "TASK_SERVICE", None), patch.object(srv, "_model_context_capacity", lambda m: 32768),
                        patch.object(srv, "_model_capabilities", lambda m: ["completion"])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        service = srv.TASK_SERVICE
        if service:
            service.stop_all(5)
        for p in reversed(self.patches):
            p.stop()
        self.httpd.shutdown()
        self.tmp.cleanup()

    def test_chunk_plan_leaves_room_for_persian_output(self):
        service = srv._task_service()
        _, predict, chunk = service.window({"kind": "translate", "model": "m", "target": "fa"})
        self.assertLessEqual(chunk * 1.9, predict * 0.85)          # Persian output of a full chunk fits num_predict
        _, _, same = service.window({"kind": "translate", "model": "m", "target": "en"})
        self.assertGreater(same, chunk)

    def test_default_task_window_is_8k(self):
        _, _, _ = srv._task_service().window({"kind": "summarize", "model": "m"})
        ctx, _, _ = srv._task_service().window({"kind": "translate", "model": "m", "target": "fa"})
        self.assertEqual(ctx, 8192)

    def test_truncated_answer_is_split_and_whole_page_range_translated(self):
        pdf = make_pdf(3, words=240)
        service = srv._task_service()
        task = service.create("book.pdf", pdf, {"kind": "translate", "model": "m", "target": "fa", "autostart": False, "num_ctx": 32768})
        manager = service.manager
        # force big chunks so that the fake model "overflows" and the recovery split must kick in
        params = manager.get(task["id"])["params"]
        params["max_tokens"] = 6000
        manager._set(task["id"], params=params, plan="{}", total_chunks=0)
        with manager.db.tx() as db:
            db.execute("DELETE FROM chunks WHERE task_id=?", (task["id"],))
        service.start(task["id"])
        end = time.time() + 60
        saw_live = False
        while time.time() < end:
            prog = manager.progress(task["id"])
            if prog.get("live"):
                saw_live = True
            if prog["status"] in ("completed", "partial", "failed"):
                break
            time.sleep(0.01)
        prog = manager.progress(task["id"])
        self.assertEqual(prog["status"], "completed", prog)
        self.assertTrue(any(n > FakeOllama.long_limit for n in FakeOllama.calls))     # the oversize attempt happened …
        self.assertTrue(any(n <= FakeOllama.long_limit for n in FakeOllama.calls))    # … and was redone in smaller pieces
        out = "\n".join(r["output"] for r in manager.chunk_rows(task["id"]))
        for page in (1, 2, 3):
            self.assertIn(f"عنوان صفحه {page}", out)

    def test_live_info_reports_tokens(self):
        m = tasks.TaskManager(self.tmp.name + "/x")
        m.live_tick("t", 0, 0)
        info = m.live_info("t")
        self.assertEqual(info["phase"], "waiting")
        m.live_tick("t", 0, 5)
        self.assertEqual(m.live_info("t")["phase"], "generating")
        self.assertEqual(m.live_info("t")["tokens"], 5)


if __name__ == "__main__":
    unittest.main()


class GptOssThinkTests(unittest.TestCase):
    def test_gpt_oss_gets_low_reasoning_not_false(self):
        seen = {}
        svc = task_service.TaskService(lambda: tempfile.mkdtemp(), lambda *a, **k: {"message": {"content": "x"}},
                                       lambda m: 8192, lambda m: True)
        svc._chat = lambda body, stop: seen.update(body) or {"message": {"content": "ok"}}
        svc.generator("gpt-oss:20b", 8192, 2000)([{"role": "user", "content": "hi"}])
        self.assertEqual(seen["think"], "low")
        svc.generator("qwen3:8b", 8192, 2000)([{"role": "user", "content": "hi"}])
        self.assertIs(seen["think"], False)


class ResourceSaverTests(unittest.TestCase):
    def test_prepare_unloads_other_models_but_not_task_model(self):
        calls = []
        ps = {"models": [{"name": "qwen3:8b", "size": 6 * 1024**3, "size_vram": 6 * 1024**3},
                         {"name": "bge-m3:latest", "size": 1024**3, "size_vram": 1024**3},
                         {"name": "gemma3:4b", "size": 3 * 1024**3, "size_vram": 3 * 1024**3}]}
        with patch.object(srv, "_loaded_ollama_models", lambda: ps["models"]), \
             patch.object(srv, "_ollama_post", lambda path, payload, timeout=120: calls.append((path, payload)) or {}):
            released = srv._free_resources_for_task("qwen3:8b")
        self.assertEqual(sorted(released), ["bge-m3:latest", "gemma3:4b"])
        self.assertTrue(all(p["keep_alive"] == 0 for _, p in calls))
        self.assertNotIn("qwen3:8b", [p["model"] for _, p in calls])

    def test_fit_reports_gpu_share(self):
        ps = [{"name": "qwen3:30b", "size": 20 * 1024**3, "size_vram": 9 * 1024**3}]
        with patch.object(srv, "_loaded_ollama_models", lambda: ps):
            self.assertEqual(srv._task_model_fit("qwen3:30b")["gpu_percent"], 45)
            self.assertIsNone(srv._task_model_fit("other"))

    def test_service_calls_prepare_before_start(self):
        order = []
        tmp = tempfile.mkdtemp()
        res = type("R", (), {"prepare": lambda self, m: order.append(("prepare", m)) or ["x"], "fit": lambda self, m: None})()
        svc = task_service.TaskService(lambda: tmp, lambda *a, **k: {}, lambda m: 8192, lambda m: False, lambda m: True,
                                       lambda body, stop: {"message": {"content": "ترجمه"}}, resources=res)
        task = svc.create("a.txt", ("Hello world. " * 40).encode(), {"kind": "translate", "model": "m", "autostart": True})
        svc.manager.wait(task["id"], 20)
        self.assertEqual(order[0], ("prepare", "m"))
        svc.stop_all(3)


class SaveOutputTests(unittest.TestCase):
    def test_save_writes_unique_files_into_output_dir_and_reveal_is_confined(self):
        from local_chat import settings as app_settings
        tmp = Path(tempfile.mkdtemp())
        with patch.object(app_settings, "effective_output_dir", lambda: tmp / "out"):
            first = srv._save_task_output(b"abc", "my:book?.pdf")
            second = srv._save_task_output(b"def", "my:book?.pdf")
            self.assertEqual(Path(first["path"]).read_bytes(), b"abc")
            self.assertNotEqual(first["path"], second["path"])
            self.assertTrue(Path(first["path"]).parent == (tmp / "out"))
            with self.assertRaises(task_service.ServiceError):
                srv._reveal_in_explorer(str(tmp / "elsewhere.txt"))

    def test_set_output_dir_validates(self):
        from local_chat import settings as app_settings
        tmp = Path(tempfile.mkdtemp())
        with patch.object(app_settings, "settings_path", lambda: tmp / "settings.json"):
            with self.assertRaises(ValueError):
                app_settings.set_output_dir("relative/path")
            folder = app_settings.set_output_dir(str(tmp / "mine"))
            self.assertEqual(app_settings.effective_output_dir(), folder)
            app_settings.set_output_dir("")
            self.assertEqual(app_settings.effective_output_dir(), app_settings.default_output_dir())
