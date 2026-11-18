import numpy as np

from local_chat import files as attach_files
from local_chat import knowledge, persian_speech, speech_audio
from local_chat.project_workspace import ProjectWorkspace


# ----------------------------------------------------------------------------------------------- speech text
def test_every_spoken_chunk_ends_with_a_full_stop():
    text = ("این جمله بسیار طولانی است و باید در چند تکه خوانده شود، " * 8).strip()
    chunks = persian_speech.split_for_speech(persian_speech.normalize_for_speech(text))
    assert len(chunks) >= 2
    assert all(chunk[-1] in ".!?" for chunk in chunks), chunks


def test_user_speech_words_and_unknown_word_report(tmp_path, monkeypatch):
    words = tmp_path / "speech_words.json"
    words.write_text('{"kubernetes": "کوبرنتیس"}', encoding="utf-8")
    monkeypatch.setenv("LOCALCHAT_SPEECH_WORDS", str(words))
    persian_speech.take_unknown_latin_words()
    spoken = persian_speech.normalize_for_speech("از kubernetes و frobnicator استفاده کن")
    assert "کوبرنتیس" in spoken and "kubernetes" not in spoken
    assert persian_speech.take_unknown_latin_words() == ["frobnicator"]


# ----------------------------------------------------------------------------------------------- speech audio
def _tone(seconds, sr=24000, level=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return (level * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_clean_piece_cuts_runon_mumble_and_silence():
    sr = 24000
    silence = np.zeros(int(sr * 0.4), dtype=np.float32)
    mumble = (0.1 * np.random.RandomState(1).randn(sr * 6)).astype(np.float32)
    raw = np.concatenate([silence, _tone(2.0), silence, mumble])
    cleaned = speech_audio.clean_piece(raw, sr, 28)
    assert 1.9 <= len(cleaned) / sr <= 2.6
    assert abs(float(cleaned[-1])) < 1e-3                     # faded out, no click


def test_clean_piece_keeps_normal_speech():
    sr = 24000
    raw = np.concatenate([np.zeros(sr // 4, np.float32), _tone(2.0), np.zeros(sr // 4, np.float32)])
    cleaned = speech_audio.clean_piece(raw, sr, 28)
    assert len(cleaned) / sr >= 2.0


# ----------------------------------------------------------------------------------------------- embeddings
def test_prefix_and_threshold_depend_on_the_model():
    assert knowledge.embed_prefix("nomic-embed-text:latest", "query") == "search_query: "
    assert knowledge.embed_prefix("nomic-embed-text", "document") == "search_document: "
    assert knowledge.embed_prefix("bge-m3", "query") == ""
    assert knowledge.min_similarity("nomic-embed-text") < knowledge.min_similarity("bge-m3")


def _fake_embed(texts, kind):
    # "meaning" axis 0 = money/cost words, axis 1 = anything else: no shared literal word with the query is needed
    money = ("قیمت", "هزینه", "ریال", "تومان")
    return [[1.0, 0.05] if any(word in text for word in money) else [0.05, 1.0] for text in texts]


def test_workspace_semantic_retrieval_finds_passage_without_shared_words(tmp_path):
    store = ProjectWorkspace(tmp_path / "w.sqlite3")
    filler = "این بخش دربارهٔ تاریخچهٔ شرکت و ساختار سازمانی است. " * 60
    target = "بهای نهایی کالا برابر ۵۰۰ تومان است و هزینهٔ ارسال جداگانه محاسبه می‌شود. " * 3
    record = {"id": "c" * 32, "name": "doc.txt", "kind": "document", "text": filler + target + filler}
    store.register(record)
    keyword_only = store.retrieve([record["id"]], "قیمت چقدر است", limit=1, max_chars=3000)
    assert all(item["via"] != "semantic" for item in keyword_only)
    found = store.retrieve([record["id"]], "قیمت چقدر است", limit=2, max_chars=4000,
                           embed_texts=_fake_embed, embed_model="fake")
    assert any("۵۰۰ تومان" in item["text"] for item in found)
    assert any(item["via"] == "semantic" for item in found)
    # vectors are cached: a second question does not re-embed the chunks
    calls = []
    store.retrieve([record["id"]], "قیمت", limit=1, max_chars=2000,
                   embed_texts=lambda texts, kind: (calls.append(kind), _fake_embed(texts, kind))[1], embed_model="fake")
    assert calls == ["query"]


def test_workspace_retrieval_survives_embedding_failure(tmp_path):
    store = ProjectWorkspace(tmp_path / "w.sqlite3")
    record = {"id": "d" * 32, "name": "doc.txt", "kind": "document", "text": "خطای اتصال پایگاه داده. " * 100}

    def broken(texts, kind):
        raise RuntimeError("ollama down")
    store.register(record)
    found = store.retrieve([record["id"]], "خطای اتصال", limit=2, max_chars=3000, embed_texts=broken, embed_model="fake")
    assert found and found[0]["via"] == "keyword"


# ----------------------------------------------------------------------------------------------- long attachment
def test_long_document_uses_relevant_excerpts_instead_of_head():
    text = "مقدمه. " * 3000 + "شناسهٔ مخفی پروژه QX-7731 است. " + "پایان. " * 3000
    record = {"id": "e" * 32, "name": "big.txt", "kind": "text", "size": len(text), "text": text,
              "excerpt_picker": lambda limit: "[کاراکترهای 1-2]\nشناسهٔ مخفی پروژه QX-7731 است."}
    block, clipped = attach_files.render_for_prompt(record, 2000)
    assert "QX-7731" in block and clipped is True
    plain = dict(record)
    plain.pop("excerpt_picker")
    head_block, _ = attach_files.render_for_prompt(plain, 2000)
    assert "QX-7731" not in head_block
