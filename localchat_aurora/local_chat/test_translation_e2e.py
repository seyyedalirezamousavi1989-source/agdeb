"""End-to-end: multi-page PDF -> persistent translate task -> every page translated -> valid PDF/DOCX download.
Also covers the follow-up «translate the whole document» attachment lookup and singular page phrases."""
import io
import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from local_chat import pagespec
from local_chat import server as srv
from local_chat.test_features import FeatureTestCase

try:
    from reportlab.pdfgen import canvas
except ImportError:  # pragma: no cover
    canvas = None


_LETTERS = "ابپتثجچحخد"


def fa_token(digits):
    """A unique, Persian-looking word per source word (keeps the repetition detector quiet)."""
    return "کا" + "".join(_LETTERS[int(d)] for d in digits) + "ی"


def make_pdf(pages, words=120):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for number in range(1, pages + 1):
        y = 800
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Heading of page {number}")
        y -= 30
        for row in range(max(1, words // 12)):
            line = f"MARKER{number:03d} row{row} " + " ".join(f"word{number}_{row}_{i}" for i in range(12))
            c.drawString(60, y, line)
            y -= 16
            if y < 60:
                break
        c.showPage()
    c.save()
    return buf.getvalue()


@unittest.skipIf(canvas is None, "reportlab not installed")
class TranslationE2ETests(FeatureTestCase):
    def setUp(self):
        super().setUp()
        self.prompts = []

        def ollama_post(path, payload, timeout=120):
            content = payload["messages"][-1]["content"]
            self.prompts.append(content)
            body = content.split("متن برای ترجمه:\n\n", 1)[1]
            def fa(line):
                if re.match(r"^\s*(#|\||```|\[\^|$)", line):
                    return line
                line = re.sub(r"word(\d+)_(\d+)_(\d+)", lambda m: fa_token(m.group(1) + m.group(2) + m.group(3)), line)
                line = re.sub(r"row(\d+)", r"ردیف \1", line)
                return line.replace("Heading of page", "عنوان صفحه")
            out = "\n".join(fa(ln) for ln in body.split("\n"))
            return {"message": {"content": out + "\n<<TERMS>>\nword = کلمه"}}

        self.patches2 = [
            patch.object(srv, "_ollama_post", ollama_post),
            patch.object(srv, "_ollama_chat_cancellable", lambda body, stop: ollama_post("/api/chat", body)),
            patch.object(srv, "APP_DATA_DIR", srv.Path(self.tmp.name)),
            patch.object(srv, "TASK_SERVICE", None),
            patch.object(srv, "_model_context_capacity", lambda model: 4096),
            patch.object(srv, "_model_capabilities", lambda model: ["completion"]),
        ]
        for item in self.patches2:
            item.start()
        self.task_id = None

    def tearDown(self):
        service = srv.TASK_SERVICE
        if service is not None:
            service.stop_all(5)
        for item in reversed(self.patches2):
            item.stop()
        super().tearDown()

    def create(self, name, data, **options):
        opts = {"kind": "translate", "model": "text:8b", "target": "fa", **options}
        status, body = self.call("POST", "/api/tasks", raw=data, headers={
            "X-Filename": quote(name), "X-Task-Options": quote(json.dumps(opts)),
            "Content-Type": "application/octet-stream"})
        payload = json.loads(body)
        self.assertEqual(status, 201, payload)
        self.task_id = payload["task"]["id"]
        return payload["task"]

    def wait_completed(self, timeout=60):
        end = time.time() + timeout
        task = {}
        while time.time() < end:
            _, body = self.call("GET", f"/api/tasks/{self.task_id}")
            task = json.loads(body)["task"]
            if task["status"] in {"completed", "failed", "partial"}:
                break
            time.sleep(0.05)
        self.assertEqual(task["status"], "completed", task)
        return task

    def download(self, fmt):
        status, body = self.call("GET", f"/api/tasks/{self.task_id}/download?format={fmt}")
        self.assertEqual(status, 200, body[:300] if isinstance(body, (bytes, str)) else body)
        return body if isinstance(body, bytes) else body.encode("latin-1")

    def test_multi_page_pdf_translates_every_page_and_exports_valid_files(self):
        pages = 12
        task = self.create("report.pdf", make_pdf(pages))
        self.assertGreaterEqual(task["total"], 1)
        done = self.wait_completed()
        self.assertEqual(done["done"], done["total"])
        joined = "\n".join(self.prompts)
        for number in range(1, pages + 1):                     # no page silently skipped
            self.assertIn(f"MARKER{number:03d}", joined, f"page {number} never sent to the model")

        docx = self.download("docx")
        from docx import Document
        text = "\n".join(p.text for p in Document(io.BytesIO(docx)).paragraphs)
        for number in range(1, pages + 1):
            self.assertIn(f"MARKER{number:03d}", text, f"page {number} missing from DOCX")

        pdf = self.download("pdf")
        self.assertTrue(pdf.startswith(b"%PDF"))
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf))
        self.assertGreaterEqual(len(reader.pages), 1)

        for fmt in ("md", "txt"):
            self.assertIn("MARKER012", self.download(fmt).decode("utf-8"))

    def test_page_selection_only_sends_selected_pages(self):
        self.create("report.pdf", make_pdf(6), pages="2-3")
        self.wait_completed()
        joined = "\n".join(self.prompts)
        self.assertIn("MARKER002", joined)
        self.assertIn("MARKER003", joined)
        self.assertNotIn("MARKER001", joined)
        self.assertNotIn("MARKER004", joined)


class IntentAndFollowUpTests(unittest.TestCase):
    def test_whole_document_phrases_do_not_restrict_pages(self):
        for text in ("translate the whole document to Persian", "کل سند را به انگلیسی ترجمه کن",
                     "translate the document from first page to last page", "ترجمه کن از صفحه اول تا آخر",
                     "این فایل را کامل ترجمه کن"):
            intent = srv._large_document_task_intent(text)
            self.assertEqual(intent["kind"], "translate", text)
            self.assertNotIn("pages", intent, text)

    def test_single_page_phrases_are_restricted(self):
        self.assertEqual(pagespec.parse_request("translate the first page")["ranges"], [[1, 1]])
        self.assertEqual(pagespec.parse_request("صفحه اول را ترجمه کن")["ranges"], [[1, 1]])
        self.assertEqual(pagespec.parse_request("translate the last page")["last"], 1)
        self.assertEqual(pagespec.parse_request("فقط ده صفحه اول را ترجمه کن")["ranges"], [[1, 10]])
        self.assertIsNone(pagespec.parse_request("translate everything"))

    def test_prior_attachment_is_found_for_followup_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.source"
            source.write_bytes(b"%PDF-1.4")
            records = {
                "att-old": {"name": "old.pdf", "kind": "document", "source_path": str(source)},
                "att-img": {"name": "pic.png", "kind": "image"},
                "att-gone": {"name": "gone.pdf", "kind": "document", "source_path": str(Path(tmp) / "missing")},
            }
            recent = [
                {"role": "user", "content": "here", "attachments": ["att-old", "att-img", "att-gone"]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "translate the whole document", "attachments": []},
            ]
            with patch.object(srv.ATTACHMENTS, "get", side_effect=records.get):
                found = srv._prior_task_attachments(recent, 2)
            self.assertEqual([r["name"] for r in found], ["old.pdf"])
            with patch.object(srv.ATTACHMENTS, "get", side_effect=records.get):
                self.assertEqual(srv._prior_task_attachments(recent, 0), [])


if __name__ == "__main__":
    unittest.main()
