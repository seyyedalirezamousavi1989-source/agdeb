"""End to end: upload a site ZIP, ask about it and ask for a change in programming mode (real server, scripted Ollama)."""
import http.client
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from local_chat import files as attach_files
from local_chat import system_tools
from local_chat import test_programming_e2e as base

SITE = {
    "shop/index.html": "<html><body><nav class=menu>Menu</nav><h1>فروشگاه</h1></body></html>",
    "shop/css/style.css": ".menu { text-align: left; }\n",
    "shop/js/app.js": "console.log('app');\n",
    "shop/node_modules/x/index.js": "junk",
}


def site_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in SITE.items():
            archive.writestr(name, text)
    return buffer.getvalue()


class ProjectE2E(base.ProgrammingE2ETests):
    """Reuses the programming-mode harness (fake Ollama, server, approval helpers); the inherited tests are skipped here."""

    def setUp(self):
        super().setUp()
        from local_chat import server
        self.uploads = tempfile.TemporaryDirectory()
        self.extra = [patch.object(server, "ATTACHMENTS", attach_files.AttachmentStore(lambda: Path(self.uploads.name))),
                      patch.object(server.WORKSPACES, "register", lambda record: {"indexed": False})]
        for item in self.extra:
            item.start()

    def tearDown(self):
        for item in self.extra:
            item.stop()
        self.uploads.cleanup()
        super().tearDown()

    def upload(self, name="shop.zip", data=None):
        data = data or site_zip()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {**self.browser.headers(), "Content-Type": "application/octet-stream", "X-Filename": name,
                   "Content-Length": str(len(data))}
        conn.request("POST", "/api/upload", data, headers)
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        self.assertEqual(response.status, 201, body)
        return body["attachment"]["id"]

    def chat_with(self, attachment_id, text, history=(), on_event=None):
        messages = list(history) + [{"role": "user", "content": text, "attachments": [attachment_id] if attachment_id else []}]
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        conn.request("POST", "/api/chat", json.dumps({"model": "fake-coder:latest", "messages": messages, "mode": "programming"}),
                     self.browser.headers())
        response = conn.getresponse()
        if response.status != 200:
            self.fail(response.read()[:300])
        events, text_out = [], ""
        for line in response:
            if line.strip():
                event = json.loads(line)
                events.append(event)
                text_out += (event.get("message") or {}).get("content", "") or ""
                if on_event:
                    on_event(event)
        conn.close()
        return events, text_out

    @staticmethod
    def system_prompt_of(payload):
        return payload["messages"][0]["content"]

    def test_zip_is_imported_and_reviewed_with_real_file_reads(self):
        attachment = self.upload()
        script = iter([
            {"tool_calls": [("list_files", {"project": "shop"})]},
            {"tool_calls": [("search_files", {"project": "shop", "query": "text-align"})]},
            {"content": ["منوی بالا در css/style.css:1 چپ‌چین است."]},
        ])
        self.ollama.script = lambda kind, payload: next(script)
        events, text = self.chat_with(attachment, "این سایت رو تحلیل کن")
        self.assertTrue((self.root / "shop" / "index.html").is_file())
        self.assertFalse((self.root / "shop" / "node_modules").exists())
        self.assertFalse((self.root / "shop" / ".venv").exists())
        self.assertEqual(self.ollama.kinds(), ["loop", "loop", "loop"])
        # the model got the project through tools, not as a pasted digest
        first_user = [m for m in self.ollama.requests[0][1]["messages"] if m["role"] == "user"][0]["content"]
        self.assertIn("در فضای کار وارد شد", first_user)
        self.assertNotIn("text-align", first_user)
        self.assertIn("style.css:1: .menu { text-align: left; }", json.dumps(
            [m["content"] for m in self.ollama.requests[2][1]["messages"] if m["role"] == "tool"], ensure_ascii=False))
        self.assertIn("css/style.css:1", text)
        self.assertEqual((self.root / "shop" / "css" / "style.css").read_text(encoding="utf-8"), ".menu { text-align: left; }\n")
        self.assertTrue(any("notice" in e and "وارد فضای کار شد" in e["notice"] for e in events))
        # real window + exact usage are now reported in programming mode too
        self.assertTrue(any("runtime" in e for e in events))
        self.assertTrue(any("context_measured" in e for e in events))

    def test_followup_fix_edits_the_same_project_with_approval(self):
        attachment = self.upload()
        self.ollama.script = lambda kind, payload: {"content": ["ok"]}
        history, _ = [], None
        events, text = self.chat_with(attachment, "این سایت رو تحلیل کن")
        history = [{"role": "user", "content": "این سایت رو تحلیل کن", "attachments": [attachment]},
                   {"role": "assistant", "content": "تحلیل انجام شد."}]
        plan = {"decision": "plan", "title": "راست‌چین", "summary": "منو راست‌چین شود", "project": "wrong-name", "kind": "node",
                "spec": ["منو راست‌چین"], "steps": [
                    {"title": "ساخت پروژه", "detail": "x", "tool": "create_project"},
                    {"title": "ویرایش", "detail": "css", "tool": "edit_file"}]}

        def script(kind, payload):
            if kind == "plan":
                return {"content": json.dumps(plan)}
            used = [m for m in payload["messages"] if m["role"] == "tool"]
            if not used:
                return {"tool_calls": [("edit_file", {"project": "shop", "path": "css/style.css",
                                                      "old_text": "text-align: left", "new_text": "text-align: right"})]}
            return {"content": ["انجام شد."]}
        self.ollama.script = script
        seen = []
        events, text = self.chat_with(None, "منو رو راست‌چین کن و اصلاح کن", history, self.answer_cards(auto_files=False, seen=seen))
        self.assertEqual(seen, ["edit_file"])
        self.assertEqual((self.root / "shop" / "css" / "style.css").read_text(encoding="utf-8"), ".menu { text-align: right; }\n")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["shop"], "no second project may be created")
        self.assertTrue(list((self.root / "shop" / ".history").glob("css__style.css.*.bak")))

    def test_new_site_from_scratch_is_refused_without_model_call(self):
        self.ollama.script = lambda kind, payload: {"content": "must not be called"}
        events, text = self.browser.chat("یک سایت فروشگاهی بساز")
        self.assertEqual(self.ollama.kinds(), [])
        self.assertIn("اصلاح پروژه‌های موجود", text)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_corrupt_zip_is_rejected_at_upload_and_creates_nothing(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = b"PK\x03\x04 broken"
        conn.request("POST", "/api/upload", data, {**self.browser.headers(), "Content-Type": "application/octet-stream",
                                                     "X-Filename": "bad.zip", "Content-Length": str(len(data))})
        status = conn.getresponse().status
        conn.close()
        self.assertEqual(status, 415)
        self.assertEqual(list(self.root.iterdir()), [])


_OWN = {name for name in vars(ProjectE2E) if name.startswith("test_")}
for _name in [n for n in dir(base.ProgrammingE2ETests) if n.startswith("test_") and n not in _OWN]:
    setattr(ProjectE2E, _name, None)       # run only the project tests, not the inherited programming-mode ones

if __name__ == "__main__":
    unittest.main()
