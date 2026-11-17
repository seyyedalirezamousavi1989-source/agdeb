"""Private knowledge base: ingest → keyword/vector/hybrid search → citable prompt block → manage."""
import tempfile
import time
import unittest
from pathlib import Path

from local_chat import knowledge
from local_chat.test_docengine import make_book

TOPICS = {"حافظه": [1, 0, 0], "گرافیک": [0, 1, 0], "شبکه": [0, 0, 1]}


def fake_embed(model, texts):
    out = []
    for text in texts:
        vec = [float(sum(text.count(k) for k in [word])) for word in TOPICS]
        out.append(vec if any(vec) else [0.01, 0.01, 0.01])
    return out


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = ["llama3:8b"]
        self.service = knowledge.KnowledgeService(lambda: Path(self.tmp.name) / "k.sqlite3", fake_embed, lambda: self.models)

    def tearDown(self):
        self.service.close()
        self.tmp.cleanup()

    def test_persian_keyword_search_without_embedding_model(self):
        self.service.add_text("راهنمای کارت گرافیک", "# نصب درایور\nبرای نصب درایور کارت گرافیک ابتدا نسخهٔ قدیمی را حذف کنید.\n\n# خنک‌سازی\nفن‌ها باید تمیز باشند.")
        self.service.add_text("آشپزی", "# پلو\nبرنج را سه ساعت خیس کنید و بعد آب‌کش کنید تا دانه‌دانه شود.")
        hits = self.service.search("درایور گرافیک را چطور نصب کنم؟")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["title"], "راهنمای کارت گرافیک")
        self.assertEqual(self.service.search("بورس طلا و ارز دیجیتال"), [])          # irrelevant → nothing injected

    def test_arabic_letters_and_digits_are_folded(self):
        self.service.add_text("سند", "شمارهٔ سفارش ۱۲۳۴ برای كتاب ثبت شد و ارسال گردید در تهران.")
        self.assertTrue(self.service.search("سفارش 1234 کتاب"))

    def test_collection_filter_isolation_and_validation(self):
        self.service.add_text("دستور آشپزی", "برای تهیهٔ خوراک زعفرانی باید برنج را با زعفران دم کنید.", "آشپزی")
        self.service.add_text("راهنمای شبکه", "برای پیکربندی مسیریاب، نشانی درگاه پیش‌فرض را تنظیم کنید.", "شبکه")
        self.assertTrue(self.service.search("زعفران را چطور دم کنم؟", collections=["آشپزی"]))
        self.assertEqual(self.service.search("زعفران را چطور دم کنم؟", collections=["شبکه"]), [])
        self.assertEqual(self.service.search("زعفران را چطور دم کنم؟", collections=[]), [])
        with self.assertRaises(knowledge.KnowledgeError) as ctx:
            self.service.retrieve_for_prompt("سؤال دربارهٔ غذا", collections=["وجود ندارد"])
        self.assertEqual(ctx.exception.status, 404)

    def test_profiles_reference_existing_collections_and_persist_selection(self):
        self.service.add_text("راهنمای ریسک", "مدیریت ریسک در معامله باید با اندازهٔ موقعیت و حد ضرر انجام شود.", "ریسک")
        profile = self.service.base.save_profile("مدیریت ریسک", "منابع مدیریت ریسک",
                                                  ["ریسک"], 4, {"top_k": 3, "max_chars": 2500})
        self.assertEqual(profile["collections"], ["ریسک"])
        self.assertEqual(profile["retrieval"], {"top_k": 3, "max_chars": 2500})
        self.service.base.activate_profile(profile["id"])
        self.assertTrue(next(item for item in self.service.base.list_profiles() if item["id"] == profile["id"])["active"])
        self.service.base.set_profile_enabled(profile["id"], False)
        self.assertFalse(next(item for item in self.service.base.list_profiles() if item["id"] == profile["id"])["active"])
        self.service.base.delete_profile(profile["id"])
        with self.assertRaises(knowledge.KnowledgeError) as ctx:
            self.service.base.save_profile("جعلی", "", ["وجود ندارد"])
        self.assertEqual(ctx.exception.status, 404)

    def test_hybrid_vector_finds_synonym_free_match_and_prompt_block(self):
        self.models = ["nomic-embed-text:latest", "llama3:8b"]
        self.service.add_text("حافظه", "حافظه رم موقت است و حافظه پنهان سریع‌تر است. " * 3)
        self.service.add_text("شبکه", "شبکه و مسیریاب بسته‌ها را جابه‌جا می‌کنند. " * 3)
        self.service.kick()
        for _ in range(50):
            if self.service.base.count_pending() == 0:
                break
            time.sleep(0.1)
        self.assertEqual(self.service.base.stats()["embedded"], 2)
        self.assertEqual(self.service.embedding_model(), "nomic-embed-text:latest")
        block, sources = self.service.retrieve_for_prompt("حافظه چیست")
        self.assertIn("[K1]", block)
        self.assertEqual(sources, ["حافظه"])
        self.assertIn("دستور", block.splitlines()[0])

    def test_book_upload_dedup_toggle_delete_settings(self):
        data = make_book(chapters=4, sections=2, paragraphs=3).encode()
        source = self.service.add_document("book.txt", data, "کتاب‌ها")
        self.assertGreater(source["chunks"], 1)
        with self.assertRaises(knowledge.KnowledgeError) as ctx:
            self.service.add_document("book.txt", data)
        self.assertEqual(ctx.exception.status, 409)
        self.service.base.set_enabled(source["id"], False)
        self.assertEqual(self.service.retrieve_for_prompt("chapter"), ("", []))
        self.service.base.set_enabled(source["id"], True)
        self.service.update_settings({"enabled": False})
        self.assertEqual(self.service.retrieve_for_prompt("chapter"), ("", []))
        self.service.update_settings({"enabled": True, "top_k": 3})
        self.assertTrue(self.service.base.delete(source["id"]))
        self.assertEqual(self.service.base.stats()["chunks"], 0)
        self.assertEqual(self.service.search("chapter"), [])

    def test_empty_and_bad_input(self):
        with self.assertRaises(knowledge.KnowledgeError):
            self.service.add_document("x.txt", b"")
        with self.assertRaises(knowledge.KnowledgeError):
            self.service.add_text("t", "کوتاه")
        self.assertEqual(self.service.search('"" OR ( * ) :'), [])               # hostile FTS syntax is harmless


if __name__ == "__main__":
    unittest.main()
