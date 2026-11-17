"""«Translate only 10 pages / page 5 / pages 12-30 of a big PDF» must really process only those pages."""
import io
import json
import re
import unittest
from urllib.parse import quote

from local_chat import docengine, pagespec
from local_chat import server as srv
from local_chat import test_task_api as _api


def make_pdf(pages):
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for number in range(1, pages + 1):
        word = chr(97 + number // 26) + chr(97 + number % 26) * 3          # unique per page: not mistaken for a running header
        pdf.drawString(72, 700, f"Marker{number:03d} {word} chapter text for page {number}.")
        pdf.drawString(72, 680, f"This ordinary sentence about {word}{word} fills the page with some words.")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _persian(size):
    """Non-repeating Persian filler of about ``size`` characters (a repeated sentence would be flagged as a model loop)."""
    words = ["کتاب", "فصل", "متن", "نویسنده", "داستان", "شهر", "دریا", "کوه", "باد", "نور", "خانه", "راه", "سفر", "یاد", "شب", "روز"]
    out, i = [], 0
    while sum(len(w) + 1 for w in out) < size:
        out.append(words[i % len(words)] + str(i // len(words)).translate({48 + d: 0x6F0 + d for d in range(10)}))
        i += 1
    return " ".join(out) + "."


class ParseTests(unittest.TestCase):
    def test_sentences(self):
        cases = {
            "فقط ده صفحه اول را ترجمه کن": {"ranges": [[1, 10]], "last": 0},
            "ده صفحهٔ اول": {"ranges": [[1, 10]], "last": 0},
            "فقط ۱۵ صفحه از کتاب رو ترجمه کن": {"ranges": [[1, 15]], "last": 0},
            "صفحه ۱۲ تا ۳۰ را به فارسی ترجمه کن": {"ranges": [[12, 30]], "last": 0},
            "از صفحهٔ ۵ تا صفحهٔ ۹ ترجمه کن": {"ranges": [[5, 9]], "last": 0},
            "translate pages 3-7 please": {"ranges": [[3, 7]], "last": 0},
            "صفحه ۱۷ را ترجمه کن": {"ranges": [[17, 17]], "last": 0},
            "صفحه‌های ۳ و ۷ و ۹": {"ranges": [[3, 3], [7, 7], [9, 9]], "last": 0},
            "آخرین ۴ صفحه را خلاصه کن": {"ranges": [], "last": 4},
            "translate the first 20 pages": {"ranges": [[1, 20]], "last": 0},
            "صفحه ۱ تا ۵ و آخر ۲ صفحه": {"ranges": [[1, 5]], "last": 0},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                got = pagespec.parse_request(text)
                self.assertIsNotNone(got, text)
                self.assertEqual(got["ranges"], expected["ranges"])
                if expected["last"]:
                    self.assertEqual(got["last"], expected["last"])

    def test_no_restriction_in_ordinary_requests(self):
        for text in ("این کتاب را ترجمه کن", "کتاب ۳۰۰ صفحه‌ای را به فارسی ترجمه کن", "ترجمه کن", "summarize it"):
            with self.subTest(text=text):
                self.assertIsNone(pagespec.parse_request(text))

    def test_form_list_and_resolve(self):
        spec = pagespec.parse_list("۱-3, 10 ,20 تا 22, آخر 2")
        self.assertEqual(pagespec.resolve(spec, 50), [1, 2, 3, 10, 20, 21, 22, 49, 50])
        self.assertIsNone(pagespec.parse_list("  "))
        with self.assertRaises(pagespec.PageSpecError):
            pagespec.parse_list("abc")
        with self.assertRaises(pagespec.PageSpecError):
            pagespec.resolve(pagespec.parse_list("99"), 10)
        self.assertEqual(pagespec.describe(spec), "1-3, 10, 20-22, آخرین 2")

    def test_intent_carries_pages_only_when_present(self):
        self.assertEqual(srv._large_document_task_intent("ترجمه کن"), {"kind": "translate", "target": "fa"})
        self.assertEqual(srv._large_document_task_intent("فقط ده صفحه اول را ترجمه کن")["pages"]["ranges"], [[1, 10]])


class PdfLoadTests(unittest.TestCase):
    def test_only_selected_pages_are_extracted(self):
        data = make_pdf(30)
        doc = docengine.load_document(data, "book.pdf", page_spec=pagespec.parse_list("3-4, 20"))
        text = doc.markdown()
        for wanted in (3, 4, 20):
            self.assertIn(f"Marker{wanted:03d}", text)
        for unwanted in (1, 2, 5, 19, 21, 30):
            self.assertNotIn(f"Marker{unwanted:03d}", text)
        self.assertTrue(any("3-4, 20" in note and "30" in note for note in doc.notes))
        self.assertEqual({b.page for b in doc.blocks}, {3, 4, 20})

    def test_errors_are_clear(self):
        with self.assertRaises(docengine.DocumentError) as ctx:
            docengine.load_document(make_pdf(5), "b.pdf", page_spec=pagespec.parse_list("9"))
        self.assertIn("5 صفحه", str(ctx.exception))
        with self.assertRaises(docengine.DocumentError):
            docengine.load_document(b"hello world text", "a.txt", page_spec=pagespec.parse_list("1-2"))


class PageTaskApiTests(_api.TaskApiTests):
    """Reuses the fake-Ollama API harness (its own inherited tests run again; that is fine and cheap)."""

    def test_task_translates_only_requested_pages(self):
        from unittest.mock import patch
        seen = []

        def spy(path, payload, timeout=120):
            body = payload["messages"][-1]["content"]
            seen.append(body)
            return {"message": {"content": _persian(len(body) // 3)}}
        with patch.object(srv, "_ollama_chat_cancellable", lambda body, stop: spy("/api/chat", body)):
            status, payload = self.create(name="big.pdf", data=make_pdf(40), pages="5-6")
            self.assertEqual(status, 201, payload)
            task = self.wait_status({"completed", "partial", "failed"})
            self.assertEqual(task["status"], "completed", srv._task_service().detail(self.task_id)["events"])
        self.assertIn("(صفحه 5-6)", task["title"])
        joined = "\n".join(seen)
        self.assertIn("Marker005", joined)
        self.assertIn("Marker006", joined)
        self.assertNotIn("Marker007", joined)
        self.assertNotIn("Marker001", joined)

    def test_chat_request_for_ten_pages_creates_a_page_limited_task(self):
        from unittest.mock import patch
        status, info = self.upload("book.pdf", make_pdf(40))
        self.assertEqual(status, 201, info)
        with patch.object(srv, "_ollama_chat_cancellable", lambda body, stop: {"message": {"content": _persian(300)}}):
            status, body = self.chat("text:8b", "فقط ده صفحه اول این کتاب را ترجمه کن", [info["attachment"]["id"]])
            self.assertEqual(status, 200)
            self.assertIn("فقط صفحه‌های 1-10", body.decode())
            _, listing = self.call("GET", "/api/tasks")
            tasks = json.loads(listing)["tasks"]
            self.assertEqual(len(tasks), 1)
            self.assertIn("(صفحه 1-10)", tasks[0]["title"])
            self.task_id = tasks[0]["id"]
            self.wait_status({"completed", "partial"})

    def test_chat_page_request_on_docx_style_file_is_explained(self):
        status, info = self.upload("notes.txt", ("Plain text sentence. " * 80).encode())
        status, body = self.chat("text:8b", "فقط ده صفحه اول را ترجمه کن", [info["attachment"]["id"]])
        self.assertIn("فقط برای فایل PDF", body.decode())
        _, listing = self.call("GET", "/api/tasks")
        self.assertEqual(json.loads(listing)["tasks"], [])

    def test_bad_page_requests_are_rejected_before_creating_a_task(self):
        status, payload = self.create(name="big.pdf", data=make_pdf(5), pages="50")
        self.assertEqual(status, 415)
        self.assertIn("5 صفحه", payload["error"])
        status, payload = self.create(name="n.txt", data=b"hello there. " * 50, pages="1-2")
        self.assertEqual(status, 415)
        status, payload = self.create(name="big.pdf", data=make_pdf(5), pages="xyz")
        self.assertEqual(status, 400)
        _, body = self.call("GET", "/api/tasks")
        self.assertEqual(json.loads(body)["tasks"], [])


for _name in [n for n in dir(PageTaskApiTests) if n.startswith("test_") and n not in PageTaskApiTests.__dict__]:
    setattr(PageTaskApiTests, _name, None)          # do not re-run the inherited API tests here


if __name__ == "__main__":
    unittest.main()
