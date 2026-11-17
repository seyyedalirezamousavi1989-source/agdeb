# LocalChat 2.0.16 — Adaptive Context (Phase 19)

## Root causes found in 2.0.15
1. `hardware.kv_bytes_per_token` multiplied **all** layers (65 for qwen35) by KV heads; per-layer lists used `max()`. For hybrid models only ~every 4th layer has a KV cache → ≈3.8× over-estimate (66 GB vs ≈17.8 GB at 262144 for qwen3.8:27b).
2. The window was recomputed from VRAM readings that differ between cold/loaded state → `num_ctx` could flip between requests and make Ollama reload the model.
3. Nothing compared the requested `num_ctx` with Ollama's real `context_length` (`/api/ps`); a `Modelfile` `PARAMETER num_ctx 8192` or another client's load was invisible.
4. No GPU telemetry ⇒ RAM ignored entirely; "Ollama down" and "nothing loaded" were both `[]`.
5. Programming mode used a fixed size table (8k/12k/16k) independent of hardware; "make room" could grow past what memory allowed.
6. Prompt char budget was always `tokens × 2.1` (Persian worst case): English/code chats dropped history at ~55 % of the window.
7. `prompt_eval_count` / `eval_count` from Ollama were never used; all token numbers were estimates without being labelled so.

## New / changed modules (local_chat/)
- `context_arch.py` (new) – architecture adapters: `hybrid_recurrent` (full-attention layers by `full_attention_interval` or a per-layer KV-head list + fixed recurrent state from `ssm.*`), `full_attention` (MHA/GQA; sliding-window / latent-KV flagged `upper_bound`), `unsupported` (never guesses).
- `model_inspector.py` (new) – `/api/show` + `/api/tags` → validated `ModelProfile` (declared context, Modelfile num_ctx, issues list), cached.
- `runtime_inspector.py` (new) – `/api/ps` parsing, `ContextTriple` (declared / requested / actual, status verified|mismatch|unknown|not_loaded|unreachable).
- `adaptive_context.py` (new) – estimator, calibration (only raises estimates), allocator (largest window that fits GPU+RAM and keeps estimated decode slowdown ≤ 1.25×), `ContextCoordinator` (reload policy: keep resident window, grow once when a request needs >60 %, shrink only when over limits, no conflicting concurrent loads).
- `hardware.py` – `snapshot()`; RAM without psutil; GPU via NVML → nvidia-smi → Windows registry/counters → Linux sysfs; integrated GPUs never counted as VRAM; unmeasurable values are `None` + listed in `unavailable`. `pick_window` kept for compatibility (no longer used by the server).
- `context_manager.py` – `TokenCalibrator`, `measured_usage`, `adapt_char_budget`, `recovery_report`, `guard_request(token_ratio=)`.
- `server.py` – wiring; `/api/runtime/apply-context` (explicit reload + verification); runtime events `runtime` and `context_measured`; status panel shows declared/actual/plan.
- `settings.py`, `static/app.js`, `static/runtime_panel.js` – ceiling choices up to 262144 and "auto" (0); compact diagnostics line; apply-window button.
- `test_adaptive_context.py` (new, 62 tests); one existing test adjusted (it assumed the last urlopen call was `/api/chat`).

## Known limits
- All memory figures are **estimates**; only `/api/ps` values and Ollama's token counts are measured. Real-hardware behaviour of qwen3.8:27b must still be validated with incremental window increases.
- KV cache type / parallel slots are read from LocalChat's own environment (assumption); calibration corrects an under-estimate after the model is loaded.
- Windows registry/counter GPU discovery and NVML on Windows are unit-tested with fixtures only, not on a real Windows machine.
- Pure SSM models and unknown architectures use labelled fallbacks.
- `prompt_eval_count` may exclude cached prompt tokens, so it is only used to raise the estimate.

---
## نسخه 2.0.17 — مدل‌های آنلاین (Phase 20)
- مدیریت مدل‌ها ← تب «آنلاین ☁»: ثبت/جایگزینی/حذف کلید API، فهرست مدل‌ها، دکمهٔ «استفاده در گفتگو».
- نام مدل: `online:anthropic/<id>`؛ کلید در settings.json (ویندوز: DPAPI؛ سایر سیستم‌ها: base64 با دسترسی 0600) و هرگز به مرورگر برگردانده نمی‌شود.
- پاسخ SSE ی Anthropic به NDJSON هم‌شکل Ollama تبدیل می‌شود؛ `prompt_eval_count/eval_count` از usage می‌آید.
- محدودیت‌ها: ابزارها (وب/فارکس) برای مدل آنلاین ارسال نمی‌شوند و خروجی ابزار به‌صورت متن به زمینه اضافه می‌شود؛ تماس واقعی با Anthropic در محیط ساخت آزمون نشد (تست‌ها با شبیه‌ساز).
