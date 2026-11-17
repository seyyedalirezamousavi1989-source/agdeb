"""Two embedding models side by side (nomic-embed-text 768-d style, bge-m3 1024-d style) with a deterministic fake embedder."""
import hashlib
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

try:
    from . import knowledge
except ImportError:  # pragma: no cover
    import knowledge

DIMS = {"nomic-embed-text:latest": 8, "bge-m3:latest": 12}
TEXTS = {
    "fa": "گربه‌ها حیوانات خانگی دوست‌داشتنی هستند و بیشتر روز را می‌خوابند. " * 3,
    "en": "Solar panels convert sunlight into electricity using photovoltaic cells in rooftop arrays. " * 3,
}


def fake_embed(model, texts):
    dim = DIMS[model]
    out = []
    for text in texts:
        seed = hashlib.sha256(text.encode()).digest()
        out.append([(seed[i % 32] + 1) / 256.0 for i in range(dim)])
    return out


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.installed = list(DIMS)
        self.embed = fake_embed
        self.service = knowledge.KnowledgeService(
            lambda: Path(self.tmp.name) / "k.sqlite3", lambda m, t: self.embed(m, t), lambda: self.installed)

    def tearDown(self):
        self.service.close(3)
        self.tmp.cleanup()

    def fill(self, model):
        self.service.update_settings({"embed_model": model})
        self.service.kick()
        end = time.time() + 10
        while time.time() < end and self.service.base.count_pending(model):
            time.sleep(0.05)
        self.assertEqual(self.service.base.count_pending(model), 0)

    def add_docs(self):
        self.service.base.add_text("گربه", TEXTS["fa"], "fa")
        self.service.base.add_text("Solar", TEXTS["en"], "en")

    def test_both_models_index_with_their_real_dimensions(self):
        self.add_docs()
        self.fill("nomic-embed-text:latest")
        self.fill("bge-m3:latest")
        cov = {c["model"]: c["dim"] for c in self.service.base.vector_coverage()}
        self.assertEqual(cov, DIMS)

    def test_switching_model_keeps_the_other_index(self):
        self.add_docs()
        self.fill("nomic-embed-text:latest")
        before = self.service.base.stats("nomic-embed-text:latest")["embedded"]
        result = self.service.update_settings({"embed_model": "bge-m3:latest"})
        self.assertTrue(result["model_changed"])
        self.assertEqual(result["coverage"]["embedded"], 0)               # new model not indexed yet: reported, not hidden
        self.assertEqual(self.service.base.stats("nomic-embed-text:latest")["embedded"], before)

    def test_search_never_queries_another_models_vectors(self):
        self.add_docs()
        self.fill("nomic-embed-text:latest")
        self.service.base.update_settings({"embed_model": "bge-m3:latest"})
        calls = []
        self.embed = lambda m, t: (calls.append(m), fake_embed(m, t))[1]
        self.service.search("solar electricity")                           # active model has no index: keyword only
        self.assertEqual(calls, [])

    def test_dimension_mismatch_is_rejected(self):
        self.add_docs()
        self.fill("bge-m3:latest")
        cid = self.service.base.pending_embeddings("nomic-embed-text:latest", 1)[0][0]
        with self.assertRaises(knowledge.KnowledgeError):
            self.service.base.store_embeddings("bge-m3:latest", [(cid, [0.1] * 7)])

    def test_invalid_vectors_are_rejected(self):
        v = knowledge.KnowledgeBase.validate_vectors
        for bad in ([[]], [[float("nan")] * 4], [[0.0] * 4], [["x"] * 4], ["notalist"], [[1, 2], [1, 2, 3]]):
            with self.assertRaises(knowledge.KnowledgeError):
                v(bad)
        self.assertEqual(v([[0.1, 0.2, 0.3]]), 3)

    def test_reindex_is_non_destructive_and_resumable(self):
        self.add_docs()
        self.fill("bge-m3:latest")
        base = self.service.base
        base.request_reindex("bge-m3:latest")
        self.assertEqual(base.stats("bge-m3:latest")["embedded"], 2)       # old vectors still searchable
        self.assertEqual(base.count_pending("bge-m3:latest"), 2)           # but queued for replacement
        batch = base.pending_embeddings("bge-m3:latest", 1)                # "interrupt" after one chunk
        base.store_embeddings("bge-m3:latest", [(batch[0][0], fake_embed("bge-m3:latest", ["x"])[0])])
        self.assertEqual(base.count_pending("bge-m3:latest"), 1)

    def test_legacy_single_slot_vectors_are_migrated_not_lost(self):
        path = Path(self.tmp.name) / "legacy.sqlite3"
        base = knowledge.KnowledgeBase(path)
        base.add_text("Solar", TEXTS["en"], "en")
        cid = base.pending_embeddings("x", 1)[0][0]
        vec = fake_embed("bge-m3:latest", ["a"])[0]
        with sqlite3.connect(path) as db:                                   # simulate a v2 row
            db.execute("DELETE FROM chunk_vectors")
            db.execute("UPDATE chunks SET embedding=?, emb_model='bge-m3:latest' WHERE id=?", (knowledge._pack(vec), cid))
        reopened = knowledge.KnowledgeBase(path)
        self.assertEqual(reopened.model_dimension("bge-m3:latest"), 12)
        self.assertEqual(reopened.stats("bge-m3:latest")["embedded"], 1)

    def test_test_embedding_reports_unavailable_and_ok(self):
        self.installed = ["bge-m3:latest"]
        bad = self.service.test_embedding("nomic-embed-text:latest")
        self.assertFalse(bad["ok"]); self.assertFalse(bad["installed"])
        ok = self.service.test_embedding("bge-m3:latest")
        self.assertTrue(ok["ok"]); self.assertEqual(ok["dimension"], 12)

    def test_invalid_embedding_response_is_reported(self):
        self.embed = lambda m, t: [[]]
        result = self.service.test_embedding("bge-m3:latest")
        self.assertFalse(result["ok"])
        self.embed = lambda m, t: []
        self.assertFalse(self.service.test_embedding("bge-m3:latest")["ok"])

    def test_collection_filter_applies_to_vector_search(self):
        self.add_docs()
        self.fill("bge-m3:latest")
        hits = self.service.search("Solar panels convert sunlight into electricity", collections=["fa"])
        self.assertTrue(all(h["title"] != "Solar" for h in hits))


if __name__ == "__main__":
    unittest.main()
