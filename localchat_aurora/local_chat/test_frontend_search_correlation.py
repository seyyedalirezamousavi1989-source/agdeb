"""The browser must accept a search result whose request id matches the message that asked for it.

Regression: searchWeb() returned no requestId, so the caller's `userMessage.requestId !== found.requestId`
check was always true and every successful search was discarded with «نتیجهٔ جست‌وجو متعلق به پیام فعلی نیست»."""
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

APP_JS = Path(__file__).with_name("static") / "app.js"


@unittest.skipUnless(shutil.which("node"), "node not installed")
class SearchCorrelationTests(unittest.TestCase):
    def run_node(self, server_reply):
        source = APP_JS.read_text(encoding="utf-8")
        match = re.search(r"async function searchWeb\(.*?\n}\n", source, re.S)
        self.assertIsNotNone(match, "searchWeb() not found")
        script = f"""
        const reply = {json.dumps(server_reply)};
        async function apiFetch() {{ return {{ ok: reply.ok !== false, json: async () => reply.body }}; }}
        {match.group(0)}
        (async () => {{
          const userMessage = {{ requestId: "req-aaaaaaaa-1111" }};
          try {{
            const found = await searchWeb("query", undefined, userMessage.requestId);
            const mismatch = userMessage.requestId !== found.requestId;   // the exact check used by sendMessage()
            console.log(JSON.stringify({{ mismatch, count: found.results.length, requestId: found.requestId }}));
          }} catch (e) {{ console.log(JSON.stringify({{ error: e.message }})); }}
        }})();
        """
        # Run from a UTF-8 file and read UTF-8 back: on Windows the default console codec (cp1252) mangles Persian text.
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "case.js"
            script_path.write_text(script, encoding="utf-8")
            out = subprocess.run(["node", str(script_path)], capture_output=True, timeout=20)
        stdout = (out.stdout or b"").decode("utf-8", errors="replace")
        self.assertEqual(out.returncode, 0, (out.stderr or b"").decode("utf-8", errors="replace"))
        return json.loads(stdout.strip().splitlines()[-1])

    def test_matching_result_is_accepted(self):
        got = self.run_node({"body": {"request_id": "req-aaaaaaaa-1111", "requested_query": "query",
                                      "results": [{"url": "https://example.org/a", "title": "A"}]}})
        self.assertEqual(got, {"mismatch": False, "count": 1, "requestId": "req-aaaaaaaa-1111"})

    def test_result_for_another_request_is_rejected(self):
        got = self.run_node({"body": {"request_id": "req-OTHER-2222222", "requested_query": "query", "results": []}})
        self.assertIn("error", got)

    def test_result_for_another_query_is_rejected(self):
        got = self.run_node({"body": {"request_id": "req-aaaaaaaa-1111", "requested_query": "stale", "results": []}})
        self.assertIn("error", got)

    def test_provider_error_message_is_preserved(self):
        got = self.run_node({"ok": False, "body": {"error": "ارائه‌دهنده پاسخ نداد"}})
        self.assertEqual(got["error"], "ارائه‌دهنده پاسخ نداد")


if __name__ == "__main__":
    unittest.main()
