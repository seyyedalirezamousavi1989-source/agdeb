"""Phone / PWA access: login, device cookies, proxy handling, blocked PC-level endpoints, PWA files."""
import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat import remote_access
from local_chat.server import ChatHandler, ThreadingHTTPServer, STATIC_DIR, APP_VERSION

PHONE_HOST = "192.168.1.50:8000"


class EvaluateRules(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(remote_access.classify("127.0.0.1", {"Host": "127.0.0.1:8000"}), "local")
        self.assertEqual(remote_access.classify("::1", {"Host": "localhost:8000"}), "local")
        self.assertEqual(remote_access.classify("::ffff:127.0.0.1", {"Host": "localhost"}), "local")
        self.assertEqual(remote_access.classify("192.168.1.9", {"Host": "192.168.1.50:8000"}), "remote")
        # a reverse proxy / Tailscale Serve connects from loopback: it must NOT count as the PC itself
        self.assertEqual(remote_access.classify("127.0.0.1", {"Host": "pc.tail1234.ts.net", "X-Forwarded-For": "100.1.2.3"}), "remote")
        self.assertEqual(remote_access.classify("127.0.0.1", {"Host": "localhost", "Tailscale-User-Login": "a@b"}), "remote")
        self.assertEqual(remote_access.classify("127.0.0.1", {"Host": "evil.example"}), "remote")

    def test_host_allowlist(self):
        for ok in ("192.168.1.5", "10.0.0.2", "100.101.102.103", "my-pc", "pc.tail1234.ts.net", "phone.local"):
            self.assertTrue(remote_access.host_allowed_for_remote(ok), ok)
        for bad in ("", "8.8.8.8", "evil.example.com", "attacker.com"):
            self.assertFalse(remote_access.host_allowed_for_remote(bad), bad)

    def test_pairing_code_shape(self):
        code = remote_access.generate_code()
        self.assertRegex(code, r"^[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}$")


class RemoteServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": cls.tmp.name, "LOCALCHAT_REMOTE": "1"})
        cls.env.start()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.env.stop()
        cls.tmp.cleanup()

    def setUp(self):
        remote_access.reset_rate_limit()

    def request(self, method, path, headers=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        out = (resp.status, dict(resp.getheaders()), raw)
        conn.close()
        return out

    def phone(self, method, path, cookie=None, extra=None, body=None):
        h = {"Host": PHONE_HOST, "X-Forwarded-For": "192.168.1.9", "Accept": "text/html" if method == "GET" and not path.startswith("/api") else "*/*"}
        if cookie:
            h["Cookie"] = cookie
        h.update(extra or {})
        return self.request(method, path, h, body)

    def login(self):
        code = remote_access.pairing_code()
        status, headers, _ = self.phone("POST", "/api/remote/login", body={"code": code, "name": "iPhone test"},
                                        extra={"Origin": "http://" + PHONE_HOST})
        self.assertEqual(status, 200)
        return headers["Set-Cookie"].split(";")[0]

    def test_pc_itself_is_unaffected(self):
        status, _, _ = self.request("GET", "/api/status", {"Host": f"127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        status, _, body = self.request("GET", "/phone", {"Host": f"127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertIn("PWA", body.decode())
        status, _, body = self.request("GET", "/api/remote/status", {"Host": f"127.0.0.1:{self.port}"})
        info = json.loads(body)
        self.assertTrue(info["enabled"])
        self.assertEqual(info["code"], remote_access.pairing_code())

    def test_phone_must_log_in(self):
        status, headers, _ = self.phone("GET", "/")
        self.assertEqual((status, headers.get("Location")), (302, "/remote-login"))
        status, _, _ = self.phone("GET", "/api/status")
        self.assertEqual(status, 401)
        status, _, _ = self.phone("GET", "/api/conversations")
        self.assertEqual(status, 401)
        for public in ("/remote-login", "/manifest.webmanifest", "/sw.js", "/apple-touch-icon.png", "/static/icons/icon-512.png", "/offline.html"):
            status, _, _ = self.phone("GET", public)
            self.assertEqual(status, 200, public)

    def test_wrong_code_then_lockout(self):
        for _ in range(remote_access.MAX_FREE_FAILS - 1):
            status, _, _ = self.phone("POST", "/api/remote/login", body={"code": "AAAA-AAAA-AAAA"})
            self.assertEqual(status, 401)
        status, headers, _ = self.phone("POST", "/api/remote/login", body={"code": "AAAA-AAAA-AAAA"})
        self.assertEqual(status, 429)           # the fifth miss starts the lock
        status, _, _ = self.phone("POST", "/api/remote/login", body={"code": remote_access.pairing_code()})
        self.assertEqual(status, 429)           # even the right code waits out the lock
        remote_access.reset_rate_limit()

    def test_login_flow_and_blocked_endpoints(self):
        cookie = self.login()
        status, _, _ = self.phone("GET", "/api/status", cookie=cookie)
        self.assertEqual(status, 200)
        status, _, body = self.phone("GET", "/", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertIn(b"manifest.webmanifest", body)
        origin = {"Origin": "http://" + PHONE_HOST}
        for blocked in ("/api/shutdown", "/api/open-url", "/api/system-tools/open-folder", "/api/setup/install", "/api/settings/models-dir", "/api/forex/chart"):
            status, _, _ = self.phone("POST", blocked, cookie=cookie, extra=origin, body={})
            self.assertEqual(status, 403, blocked)
        status, _, _ = self.phone("GET", "/api/setup/status", cookie=cookie)
        self.assertNotEqual(status, 403)         # reading Setup Center status is fine; acting is not
        # pairing admin stays on the PC
        for path in ("/phone", "/api/remote/status"):
            status, _, _ = self.phone("GET", path, cookie=cookie)
            self.assertEqual(status, 403, path)
        status, _, _ = self.phone("POST", "/api/remote/regenerate", cookie=cookie, extra=origin, body={})
        self.assertEqual(status, 403)

    def test_cross_site_posts_rejected_even_with_cookie(self):
        cookie = self.login()
        status, _, _ = self.phone("POST", "/api/search", cookie=cookie, extra={"Origin": "https://evil.example"}, body={"query": "x"})
        self.assertEqual(status, 403)
        status, _, _ = self.phone("POST", "/api/search", cookie=cookie, extra={"Origin": "http://" + PHONE_HOST, "Sec-Fetch-Site": "cross-site"}, body={"query": "x"})
        self.assertEqual(status, 403)

    def test_proxied_request_from_loopback_is_not_trusted(self):
        # Tailscale Serve / reverse proxy: TCP peer is 127.0.0.1 but the request is remote
        status, _, _ = self.request("GET", "/api/status", {"Host": "pc.tail1.ts.net", "X-Forwarded-For": "100.64.0.9"})
        self.assertEqual(status, 401)
        status, _, _ = self.request("POST", "/api/shutdown", {"Host": "localhost", "X-Forwarded-For": "100.64.0.9"}, {})
        self.assertIn(status, (401, 403))

    def test_foreign_host_header_rejected(self):
        status, _, _ = self.phone("GET", "/api/status", extra={"Host": "attacker.example.com"})
        self.assertEqual(status, 403)

    def test_revoke_and_regenerate_invalidate_cookie(self):
        cookie = self.login()
        self.assertEqual(self.phone("GET", "/api/status", cookie=cookie)[0], 200)
        remote_access.regenerate_code(revoke_devices=True)
        self.assertEqual(self.phone("GET", "/api/status", cookie=cookie)[0], 401)

    def test_cookie_attributes_over_https_proxy(self):
        code = remote_access.pairing_code()
        _, headers, _ = self.phone("POST", "/api/remote/login", body={"code": code},
                                   extra={"X-Forwarded-Proto": "https", "Origin": "http://" + PHONE_HOST})
        cookie = headers["Set-Cookie"]
        for flag in ("HttpOnly", "SameSite=Strict", "Secure"):
            self.assertIn(flag, cookie)


class RemoteDisabled(unittest.TestCase):
    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": tmp}, clear=False):
            os.environ.pop("LOCALCHAT_REMOTE", None)
            os.environ.pop("LOCALCHAT_BIND", None)
            self.assertFalse(remote_access.is_enabled())
            self.assertEqual(remote_access.bind_host(), "127.0.0.1")
            decision = remote_access.evaluate("192.168.1.9", {"Host": "192.168.1.50:8000"}, "/api/status", "GET", False)
            self.assertEqual((decision["action"], decision["status"]), ("deny", 403))
            os.environ["LOCALCHAT_REMOTE"] = "1"
            self.assertEqual(remote_access.bind_host(), "0.0.0.0")


class PwaFiles(unittest.TestCase):
    def test_manifest_and_icons(self):
        manifest = json.loads((STATIC_DIR / "manifest.webmanifest").read_text(encoding="utf-8"))
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(manifest["start_url"], "/")
        for icon in manifest["icons"]:
            self.assertTrue((STATIC_DIR / icon["src"].replace("/static/", "")).is_file(), icon["src"])
        self.assertTrue((STATIC_DIR / "icons" / "apple-touch-icon.png").is_file())

    def test_index_has_ios_tags(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for needle in ('rel="manifest"', 'rel="apple-touch-icon"', "apple-mobile-web-app-capable", "viewport-fit=cover"):
            self.assertIn(needle, html)

    def test_no_inline_styles_or_scripts_in_new_pages(self):
        # the server's CSP is style-src/script-src 'self': inline style="" attributes or <script> bodies would be blocked
        import re
        for name in ("remote_login.html", "phone.html", "offline.html"):
            html = (STATIC_DIR / name).read_text(encoding="utf-8")
            self.assertNotRegex(html, r'\sstyle\s*=', name)
            self.assertNotRegex(html, r"<script(?![^>]*\ssrc=)[^>]*>", name)

    def test_http_lan_shims_for_secure_context_only_apis(self):
        # plain-HTTP Safari has no crypto.randomUUID / navigator.clipboard: without shims sending a message silently fails
        theme = (STATIC_DIR / "theme.js").read_text(encoding="utf-8")
        self.assertIn("crypto.randomUUID", theme)
        self.assertIn('"clipboard"', theme)
        self.assertLess(theme.index("crypto.randomUUID"), theme.index("localchat-theme"))

    def test_service_worker_never_touches_api(self):
        sw = (STATIC_DIR / "sw.js").read_text(encoding="utf-8")
        self.assertIn('startsWith("/api/")', sw)
        self.assertIn("__VERSION__", sw)


if __name__ == "__main__":
    unittest.main()
