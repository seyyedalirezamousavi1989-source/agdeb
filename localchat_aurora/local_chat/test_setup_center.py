"""Setup Center: resumable downloads, hardware-based recommendations, detection and user paths (no real network / installs)."""
import http.server
import os
import tempfile
import threading
import unittest
from pathlib import Path

from local_chat import setup_center as sc

GB = 1024 ** 3
DATA = os.urandom(900_000)


class _Handler(http.server.BaseHTTPRequestHandler):
    drop_after = None          # first response is cut after this many bytes (simulates a dropped connection)
    hits = []

    def log_message(self, *_):
        return

    def do_GET(self):
        start = 0
        header = self.headers.get("Range")
        if header:
            start = int(header.split("=")[1].split("-")[0])
        _Handler.hits.append(start)
        body = DATA[start:]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if _Handler.drop_after is not None and start == 0:
            self.wfile.write(body[:_Handler.drop_after])
            _Handler.drop_after = None
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/f.bin"
        self.tmp = Path(tempfile.mkdtemp())
        _Handler.hits = []

    def tearDown(self):
        self.server.shutdown()

    def test_resumes_after_dropped_connection(self):
        _Handler.drop_after = 300_000
        out = sc.download_file(self.url, self.tmp / "f.bin", sleep=lambda s: None)
        self.assertEqual(out.read_bytes(), DATA)
        self.assertEqual(len(_Handler.hits), 2)
        self.assertGreater(_Handler.hits[1], 0)          # second request used Range from the partial size
        self.assertFalse((self.tmp / "f.bin.part").exists())

    def test_continues_a_leftover_part_file(self):
        (self.tmp / "f.bin.part").write_bytes(DATA[:500_000])
        out = sc.download_file(self.url, self.tmp / "f.bin", sleep=lambda s: None)
        self.assertEqual(out.read_bytes(), DATA)
        self.assertEqual(_Handler.hits, [500_000])

    def test_cancel(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(sc.Cancelled):
            sc.download_file(self.url, self.tmp / "g.bin", cancel=event)


def snap(vram_gb=None, ram_gb=16):
    gpus = [] if vram_gb is None else [{"name": "NVIDIA GeForce RTX", "total": int(vram_gb * GB), "dedicated": True}]
    return {"ram": {"total": int(ram_gb * GB)}, "gpus": gpus}


class RecommendTests(unittest.TestCase):
    def test_size_parsing(self):
        self.assertEqual(sc.size_gb("≈ 6.6–7.6 GB"), 7.6)
        self.assertAlmostEqual(sc.size_gb("≈ 700 MB"), 700 / 1024)
        self.assertEqual(sc.size_gb(""), 0.0)

    def test_12gb_gpu_gets_a_fully_gpu_text_model(self):
        rec = sc.recommend(snap(12, 32))
        best = rec["groups"]["text"][0]
        self.assertEqual(best["fit"], "good")
        self.assertLessEqual(best["size_gb"], rec["machine"]["fit_gb"])
        self.assertTrue(rec["starter"])

    def test_small_gpu_never_leads_with_a_model_that_does_not_fit(self):
        rec = sc.recommend(snap(4, 16))
        self.assertNotEqual(rec["groups"]["text"][0]["fit"], "too_big")
        self.assertTrue(all(e["size_gb"] <= rec["machine"]["offload_gb"] for e in rec["starter"]))

    def test_cpu_only_machine_is_flagged_and_gets_small_models(self):
        rec = sc.recommend(snap(None, 8))
        self.assertEqual(rec["machine"]["mode"], "cpu")
        self.assertIn("CPU", rec["note"])
        self.assertLess(rec["groups"]["text"][0]["size_gb"], 5)
        self.assertEqual(rec["stt_model"], "base")

    def test_installed_models_are_marked(self):
        rec = sc.recommend(snap(12, 32), installed=["qwen3:8b"])
        self.assertTrue(any(e["installed"] for e in rec["groups"]["text"] if e["id"] == "qwen3:8b"))

    def test_bigger_gpu_gets_bigger_whisper(self):
        self.assertEqual(sc.whisper_for(sc.classify_machine(snap(12))), "medium")
        self.assertEqual(sc.whisper_for(sc.classify_machine(snap(6))), "small")


class PathsAndJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["LOCALCHAT_DATA_DIR"] = str(self.tmp / "data")
        for key in ("AIRESEARCH_STT_PYTHON", "AIRESEARCH_TTS_PYTHON", "AIRESEARCH_PARVOICE_DIR"):
            os.environ.pop(key, None)

    def tearDown(self):
        for key in ("LOCALCHAT_DATA_DIR", "AIRESEARCH_STT_PYTHON", "AIRESEARCH_TTS_PYTHON", "AIRESEARCH_PARVOICE_DIR", "LOCALCHAT_STT_MODEL"):
            os.environ.pop(key, None)

    def test_user_paths_are_validated_and_exported(self):
        exe = self.tmp / "python.exe"
        exe.write_text("x")
        sc.set_path("audio_python", str(exe))
        self.assertEqual(sc.get_paths()["audio_python"], str(exe))
        self.assertEqual(os.environ["AIRESEARCH_STT_PYTHON"], str(exe))
        with self.assertRaises(ValueError):
            sc.set_path("audio_python", str(self.tmp / "nope.exe"))
        with self.assertRaises(ValueError):
            sc.set_path("parsvoice_dir", str(self.tmp))          # no config.json / model.pth
        with self.assertRaises(ValueError):
            sc.set_path("bogus", "x")
        sc.set_path("audio_python", "")
        self.assertNotIn("audio_python", sc.get_paths())

    def test_configured_ollama_path_wins(self):
        exe = self.tmp / "ollama.exe"
        exe.write_text("x")
        sc.set_path("ollama_exe", str(exe))
        self.assertEqual(sc.find_ollama(), str(exe))

    def test_status_lists_every_component(self):
        ids = {c["id"] for c in sc.status(deep=False)["components"]}
        self.assertTrue({"ollama", "audio_env", "parsvoice", "webview2", "vc_redist"} <= ids)

    def test_job_kinds_are_whitelisted(self):
        for bad in ("rm -rf", "model:", "model:a b", "whisper:huge", "ollama; calc"):
            with self.assertRaises(ValueError):
                sc.start_job(bad)

    def test_missing_ollama_is_reported_for_chat_only_when_absent(self):
        original = (sc.find_ollama, sc.ollama_running)
        try:
            sc.find_ollama, sc.ollama_running = (lambda: ""), (lambda timeout=2: False)
            self.assertEqual(sc.missing_for("chat")["component"], "ollama")
            sc.find_ollama, sc.ollama_running = (lambda: "x"), (lambda timeout=2: True)
            self.assertIsNone(sc.missing_for("chat"))
        finally:
            sc.find_ollama, sc.ollama_running = original

    def test_jobs_resume_after_restart(self):
        sc._JOBS.clear()
        sc._jobs_file().write_text('[{"id":"a1","kind":"model:qwen3:8b","state":"running","created":1}]', encoding="utf-8")
        started = []
        original = sc.start_job
        sc.start_job = lambda kind: started.append(kind) or {"kind": kind}
        try:
            sc.resume_pending()
        finally:
            sc.start_job = original
        self.assertEqual(started, ["model:qwen3:8b"])


if __name__ == "__main__":
    unittest.main()


class NoDownloadAtBuildOrSetupTests(unittest.TestCase):
    """Building and running Setup must not download anything: every download happens later, in the app's Setup Center."""
    ROOT = Path(__file__).resolve().parents[1]

    def test_installer_script_downloads_nothing(self):
        text = (self.ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
        for forbidden in ("DownloadTemporaryFile", "vc_redist", "MicrosoftEdgeWebView2", "http://", "https://aka.ms", "go.microsoft.com"):
            self.assertNotIn(forbidden, text, forbidden)
        self.assertNotIn("PrepareToInstall", text)

    def test_build_script_has_no_unconditional_download(self):
        text = (self.ROOT / "build_installer.bat").read_text(encoding="utf-8")
        for forbidden in ("Invoke-WebRequest", "raw.githubusercontent.com", "curl ", "wget "):
            self.assertNotIn(forbidden, text, forbidden)
        # pip may run only behind the "tools missing" check, never unconditionally
        self.assertIn("goto tools_ready", text)
        self.assertLess(text.index("goto tools_ready"), text.index('pip install --upgrade --force-reinstall "numpy'))
        self.assertTrue((self.ROOT / "packaging" / "Farsi.isl").is_file())

    def test_microsoft_runtimes_are_installable_from_the_app(self):
        for kind in ("vc_redist", "webview2"):
            self.assertTrue(sc._KIND_OK.fullmatch(kind))


class SetupApiTests(unittest.TestCase):
    """The HTTP surface: status/recommend/jobs are readable, installs are gated and validated.

    Isolated from the machine it runs on: own data folder, creator check mocked both ways, and start_job stubbed so a test
    can NEVER download or install anything (a build PC with a registered creator used to start a real Ollama job here)."""

    @classmethod
    def setUpClass(cls):
        import http.client as _http
        from local_chat import server
        cls.http, cls.srv = _http, server
        cls.data = tempfile.mkdtemp()
        cls.saved_env = os.environ.get("LOCALCHAT_DATA_DIR")
        os.environ["LOCALCHAT_DATA_DIR"] = cls.data
        cls.saved = (server._registered_creator, sc.start_job)
        cls.started = []
        sc.start_job = lambda kind: cls.started.append(kind) or {"kind": kind, "state": "queued"}
        cls.server = server.ThreadingHTTPServer(("127.0.0.1", 0), server.ChatHandler)
        cls.port = cls.server.server_port
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.srv._registered_creator, sc.start_job = cls.saved
        if cls.saved_env is None:
            os.environ.pop("LOCALCHAT_DATA_DIR", None)
        else:
            os.environ["LOCALCHAT_DATA_DIR"] = cls.saved_env

    def setUp(self):
        self.started.clear()

    def request(self, method, path, body=None):
        import json
        conn = self.http.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {"Content-Type": "application/json", "Host": f"127.0.0.1:{self.port}"}
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        try:
            return response.status, json.loads(raw)
        except ValueError:
            return response.status, raw

    def test_status_recommend_and_jobs_are_served(self):
        code, data = self.request("GET", "/api/setup/status")
        self.assertEqual(code, 200)
        self.assertIn("ollama", {c["id"] for c in data["components"]})
        code, data = self.request("GET", "/api/setup/recommend")
        self.assertEqual(code, 200)
        self.assertIn("machine", data)
        self.assertIn("starter", data)
        code, data = self.request("GET", "/api/setup/jobs")
        self.assertEqual((code, isinstance(data["jobs"], list)), (200, True))

    def test_unverified_callers_cannot_install_anything(self):
        self.srv._registered_creator = lambda: None
        for path, body in (("/api/setup/install", {"kind": "ollama"}), ("/api/setup/path", {"key": "ollama_exe", "value": "C:\\x"}),
                           ("/api/setup/start-ollama", {})):
            code, _data = self.request("POST", path, body)
            self.assertEqual(code, 403, path)
        self.assertEqual(self.started, [])

    def test_verified_caller_gets_validation_but_nothing_really_installs(self):
        self.srv._registered_creator = lambda: {"name": "t", "email": "t@example.com", "role": "creator", "verified_at": "x"}
        code, _data = self.request("POST", "/api/setup/install", {"kind": "model:qwen3:8b"})
        self.assertEqual(code, 200)
        self.assertEqual(self.started, ["model:qwen3:8b"])          # reached the (stubbed) job starter
        sc.start_job, real = self.saved[1], sc.start_job             # real validation: bad kinds are 400
        try:
            for bad in ("rm -rf /", "model:", "whisper:huge", "ollama; calc"):
                code, _data = self.request("POST", "/api/setup/install", {"kind": bad})
                self.assertEqual(code, 400, bad)
        finally:
            sc.start_job = real
        self.assertEqual(self.started, ["model:qwen3:8b"])

    def test_panel_script_is_served_and_linked(self):
        conn = self.http.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/static/setup_panel.js")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b"setupChip", response.read())
        conn.request("GET", "/")
        self.assertIn(b"setup_panel.js", conn.getresponse().read())
        conn.close()


class SetupViewFrontendTests(unittest.TestCase):
    """Phase 26: the setup entry lives in the sidebar, the view replaces the chat area, and styling is CSP-safe."""
    STATIC = Path(sc.__file__).parent / "static"

    def test_button_is_in_sidebar_not_topbar(self):
        html = (self.STATIC / "index.html").read_text(encoding="utf-8")
        sidebar = html.split('<aside class="sidebar"')[1].split("</aside>")[0]
        topbar = html.split('<header class="topbar">')[1].split("</header>")[0]
        self.assertIn('id="setupChip"', sidebar)
        self.assertNotIn("setupChip", topbar)

    def test_no_javascript_injected_styles(self):
        # server CSP is style-src 'self': a <style> element created by script is blocked, so the page looked unstyled
        js = (self.STATIC / "setup_panel.js").read_text(encoding="utf-8")
        self.assertNotIn('createElement("style")', js)
        self.assertNotIn("setAttribute(\"style\"", js)
        css = (self.STATIC / "style.css").read_text(encoding="utf-8")
        for selector in (".setup-nav", ".setup-view", ".sv-hero", ".setup-banner", ".main-panel.setup-open"):
            self.assertIn(selector, css)

    def test_view_has_back_button_and_blink_states(self):
        js = (self.STATIC / "setup_panel.js").read_text(encoding="utf-8")
        css = (self.STATIC / "style.css").read_text(encoding="utf-8")
        self.assertIn("بازگشت به گفت‌وگو", js)
        self.assertIn("setup-open", js)
        for state in ("ok", "need", "busy"):
            self.assertIn(f'.setup-nav[data-state="{state}"]', css)
        self.assertIn("@keyframes setup-blink", css)

    def test_optional_components_do_not_block_green(self):
        items = {c["id"]: c for c in sc.status(deep=False)["components"]}
        for key in ("parsvoice", "vc_redist", "webview2"):
            self.assertTrue(items[key].get("optional"), key)
        self.assertFalse(items["ollama"].get("optional"))
