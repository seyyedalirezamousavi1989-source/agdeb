"""Knowledge tab API + chat injection, with the fake Ollama from test_features."""
import json
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from local_chat import knowledge
from local_chat import server as srv
from local_chat.test_features import FeatureTestCase


class KnowledgeApiTests(FeatureTestCase):
    def setUp(self):
        super().setUp()
        self.service = knowledge.KnowledgeService(lambda: Path(self.tmp.name) / "knowledge.sqlite3",
                                                  lambda model, texts: [[1.0, 0.0] for _ in texts], lambda: [])
        patcher = patch.object(srv, "KNOWLEDGE_SERVICE", self.service)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.service.close()
        super().tearDown()

    def upload(self, name, data, **options):
        status, body = self.call("POST", "/api/knowledge", raw=data, headers={
            "X-Filename": quote(name), "X-Knowledge-Options": quote(json.dumps(options)), "Content-Type": "application/octet-stream"})
        return status, json.loads(body)

    def test_upload_list_search_toggle_delete(self):
        text = "# تنظیم کاشی\nبرای نصب کاشی دیوار ابتدا سطح را تراز کنید و چسب را با ماله دندانه‌دار پخش کنید.\n".encode()
        status, body = self.upload("tile.md", text, collection="ساختمان")
        self.assertEqual(status, 201, body)
        sid = body["source"]["id"]
        self.assertEqual(self.upload("tile.md", text)[0], 409)
        status, body = self.call("GET", "/api/knowledge")
        data = json.loads(body)
        self.assertEqual(data["stats"]["sources"], 1)
        self.assertEqual(data["sources"][0]["collection"], "ساختمان")
        self.assertIn("embedding", data)
        status, body = self.call("POST", "/api/knowledge/search", {"query": "چسب کاشی دیوار"})
        self.assertEqual(json.loads(body)["results"][0]["title"], "tile")
        self.assertEqual(self.call("POST", f"/api/knowledge/{sid}/toggle", {"enabled": False})[0], 200)
        self.assertEqual(json.loads(self.call("POST", "/api/knowledge/search", {"query": "چسب کاشی دیوار"})[1])["results"], [])
        self.assertEqual(self.call("POST", f"/api/knowledge/{sid}/delete", {})[0], 200)
        self.assertEqual(self.call("POST", f"/api/knowledge/{sid}/delete", {})[0], 404)

    def test_text_note_and_bad_input(self):
        status, _ = self.call("POST", "/api/knowledge/text", {"title": "n", "text": "کوتاه"})
        self.assertEqual(status, 400)
        status, _ = self.call("POST", "/api/knowledge/text", {"title": "قانون خانه", "text": "رمز وای‌فای مهمان‌ها در یخچال نوشته شده است و هر ماه عوض می‌شود."})
        self.assertEqual(status, 201)
        self.assertEqual(self.call("POST", "/api/knowledge/settings", {"top_k": "abc"})[0], 400)
        status, body = self.call("POST", "/api/knowledge/settings", {"enabled": False, "top_k": 3})
        self.assertFalse(json.loads(body)["settings"]["enabled"])

    def test_service_handles_missing_db_directory_without_crashing(self):
        db_path = Path(self.tmp.name).parent / "nested" / "knowledge.sqlite3"
        service = knowledge.KnowledgeService(lambda: db_path, lambda model, texts: [[1.0, 0.0] for _ in texts], lambda: [])
        service.kick()
        self.addCleanup(service.close)
        time.sleep(0.2)
        self.assertIn(service.status.get("state"), {"idle", "no_model", "error"})

    def test_chat_gets_cited_excerpts_only_when_relevant(self):
        self.upload("wifi.md", "# وای‌فای\nرمز وای‌فای مهمان‌ها «پرنده۱۲۳» است و باید هر ماه عوض شود.\n".encode())
        before = len(self.ollama.chat_requests())
        status, body = self.chat("text:8b", "رمز وای‌فای مهمان‌ها چیست؟")
        self.assertEqual(status, 200)
        request = self.ollama.chat_requests()[before]
        system = request["messages"][0]["content"]
        self.assertIn("[K1]", system)
        self.assertIn("پرنده۱۲۳", system)
        self.assertIn("از دانش اختصاصی استفاده شد", body.decode())
        before = len(self.ollama.chat_requests())
        self.chat("text:8b", "سلام، حالت چطوره؟")
        self.assertNotIn("[K1]", self.ollama.chat_requests()[before]["messages"][0]["content"])
        self.service.update_settings({"enabled": False})
        before = len(self.ollama.chat_requests())
        self.chat("text:8b", "رمز وای‌فای مهمان‌ها چیست؟")
        self.assertNotIn("[K1]", self.ollama.chat_requests()[before]["messages"][0]["content"])

    def test_chat_collection_selection_restricts_retrieval(self):
        self.service.add_text("راهنمای آشپزی", "برای تهیهٔ خوراک زعفرانی باید برنج را با زعفران دم کنید.", "آشپزی")
        self.service.add_text("راهنمای شبکه", "برای پیکربندی مسیریاب، نشانی درگاه پیش‌فرض را تنظیم کنید.", "شبکه")
        before = len(self.ollama.chat_requests())
        status, body = self.chat("text:8b", "برای خوراک زعفرانی چه کار کنم؟",
                                 knowledge={"mode": "general", "collections": ["آشپزی"]})
        self.assertEqual(status, 200, body)
        system = self.ollama.chat_requests()[before]["messages"][0]["content"]
        self.assertIn("زعفران", system)
        self.assertNotIn("مسیریاب", system)

    def test_knowledge_only_returns_only_retrieved_evidence_and_never_calls_model(self):
        self.service.add_text("راهنمای وای‌فای", "رمز مهمان شبکهٔ بی‌سیم برابر با آبی‌دریاچه است و هر ماه باید تغییر کند.", "خانه")
        before = len(self.ollama.chat_requests())
        status, body = self.chat("text:8b", "رمز مهمان وای‌فای چیست؟",
                                 knowledge={"mode": "only", "collections": ["خانه"]})
        self.assertEqual(status, 200, body)
        reply = self.reply_text(self.chunks(body))
        self.assertIn("آبی‌دریاچه", reply)
        self.assertIn("شناسهٔ منبع:", reply)
        self.assertEqual(len(self.ollama.chat_requests()), before)

        status, body = self.chat("text:8b", "قیمت امروز طلا چقدر است؟",
                                 knowledge={"mode": "only", "collections": ["خانه"]})
        self.assertEqual(status, 200, body)
        self.assertIn("اطلاعات مرتبط و کافی", self.reply_text(self.chunks(body)))
        self.assertEqual(len(self.ollama.chat_requests()), before)

    def test_invalid_or_empty_selected_collection_is_reported(self):
        self.service.add_text("راهنمای شبکه", "برای پیکربندی مسیریاب نشانی درگاه پیش‌فرض را تنظیم کنید.", "شبکه")
        status, body = self.chat("text:8b", "چطور مسیریاب را تنظیم کنم؟",
                                 knowledge={"mode": "only", "collections": ["وجود ندارد"]})
        self.assertEqual(status, 404)
        self.assertIn("وجود ندارد", json.loads(body)["error"])

    def test_profile_api_and_chat_use_only_its_collections(self):
        self.service.add_text("راهنمای آشپزی", "برای تهیهٔ خوراک زعفرانی باید برنج را با زعفران دم کنید.", "آشپزی")
        self.service.add_text("راهنمای شبکه", "برای پیکربندی مسیریاب، نشانی درگاه پیش‌فرض را تنظیم کنید.", "شبکه")
        status, body = self.call("POST", "/api/knowledge/profiles", {
            "name": "آشپزی", "description": "منابع آشپزی", "collections": ["آشپزی"],
            "priority": 2, "retrieval": {"top_k": 2, "max_chars": 2500},
        })
        self.assertEqual(status, 200, body)
        profile = json.loads(body)["profile"]
        self.assertEqual(self.call("POST", f"/api/knowledge/profiles/{profile['id']}/activate", {})[0], 200)
        self.assertTrue(next(item for item in self.service.base.list_profiles() if item["id"] == profile["id"])["active"])
        before = len(self.ollama.chat_requests())
        status, body = self.chat("text:8b", "برای خوراک زعفرانی چه کار کنم؟",
                                 knowledge={"mode": "general", "profile_id": profile["id"], "collections": None})
        self.assertEqual(status, 200, body)
        system = self.ollama.chat_requests()[before]["messages"][0]["content"]
        self.assertIn("زعفران", system)
        self.assertNotIn("مسیریاب", system)

        status, body = self.chat("text:8b", "چطور مسیریاب را تنظیم کنم؟",
                                 knowledge={"mode": "only", "profile_id": profile["id"], "collections": ["شبکه"]})
        self.assertEqual(status, 400)
        self.assertIn("خارج از پروفایل", json.loads(body)["error"])
