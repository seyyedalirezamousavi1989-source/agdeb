# LocalChat audit — fixes in this build (2026-11-10)

## Fixed (root cause → change → test)
| Area | Root cause | Change | Test |
|---|---|---|---|
| Stop / stale request | `CHAT_REQUESTS` keyed by session only; a `search` request overwrote the active chat entry, so Stop found nothing and the search's cleanup deleted the chat entry; model slot stayed held | Search requests use their own key; cancel/wait match by request id (`server.py`: `_request_key`, `_find_chat_requests`) | `test_request_lifecycle.py` |
| Web search "belongs to another message" | `searchWeb()` returned no `requestId`; caller's `!==` check always failed, discarding every good result | Return `requestId`/`query` (`static/app.js`) | `test_frontend_search_correlation.py` (needs `node`) |
| Translate whole document → only first page | Follow-up message without re-attachment never created a task; model saw ~4 excerpts | `_prior_task_attachments()` finds latest supported document in earlier messages | `test_translation_e2e.py` |
| Page phrases | "first page"/"صفحه اول" returned no range (= whole doc); "first page to last page" must stay whole doc | `pagespec.py` | `test_translation_e2e.py` |
| Embedding index | One vector slot per chunk; model switch / reindex NULLed all vectors; no dimension record or validation | `chunk_vectors(chunk_id, model, dim, epoch, embedding)`; validation; non-destructive switch and re-index (epoch); `/api/knowledge/embedding/test`; clearer Ollama embed errors | `test_knowledge_models.py` |

## Migration (knowledge.sqlite3, schema 2 → 3)
Automatic on open. Existing `chunks.embedding/emb_model` are **copied** into `chunk_vectors`; old columns are left untouched and no longer used (rollback = run the previous build). Nothing is deleted. A model switch now reports coverage (`coverage`, `active_model`) and fills the new model's index in the background; search uses only the active model's vectors and falls back to keyword search until it is indexed.

## Test status (executed 2026-11-10, Python 3.12, unittest)
388 run; 0 failures; 6 import errors; 1 skipped. The 6 errors are test files that `import pytest` (pytest was not installable offline): test_audio_integration, test_conversation_store, test_media_and_version, test_packaging, test_search_engine, test_zip_intelligence. **They were not executed** — run them with `pytest local_chat` on your machine. They were also not executed in the baseline.

## NOT done / unverified
- No real Ollama was available: `nomic-embed-text` and `bge-m3` were never run. Dimensions, Persian retrieval quality, speed and memory are **unmeasured**; tests use a deterministic fake embedder. Auto-selection still prefers `bge-m3` (unchanged, based on model descriptions only). Run a benchmark on your machine before choosing a default.
- No knowledge-panel UI for model selection/test/coverage (API only).
- Persian PDF output was checked for validity, not visual shaping.
- Not audited: context budgeting, STT/TTS, SSRF/security review, performance profiling, search-provider connectivity/fallback ("connection could not be established" path).
- Stop fix is demonstrated at the registry level, not against a live Ollama stream.
