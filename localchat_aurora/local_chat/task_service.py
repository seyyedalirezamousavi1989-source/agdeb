"""HTTP-facing service for long tasks: creates/controls tasks and builds the model callable.

Kept out of ``server.py`` on purpose (that file is already huge); the server only routes to this class.
It owns no state beyond the ``TaskManager`` and takes every Ollama interaction as an injected callable.
"""
import inspect
import re
import threading
import time
import json
import logging
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import quote

try:
    from . import context_manager, docengine, docwriter, pagespec, pipeline, runtime_guard, tasks
except ImportError:                     # pragma: no cover
    import context_manager, docengine, docwriter, pagespec, pipeline, runtime_guard, tasks

KINDS = ("translate", "summarize")
MAX_TASK_UPLOAD = 128 * 1024 * 1024
MAX_PARAM_TEXT = 600
DEFAULT_TASK_CTX = 8192            # smaller window = model stays on the GPU, loads and answers much faster
MAX_WORKERS = 2                         # a local GPU rarely gains from more parallel requests
log = logging.getLogger("localchat.task_service")


class ServiceError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _clean(value, limit=MAX_PARAM_TEXT):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()[:limit]


def _is_repetition_error(message):
    text = str(message or "").casefold()
    return any(marker in text for marker in (
        "token repeat limit reached", "repetition limit reached", "repetitive output detected",
        "repetition detected", "repetition penalty limit",
    ))


def _exception_detail(exc):
    if isinstance(exc, HTTPError):
        try:
            body = exc.read(4000).decode("utf-8", "replace")
            try:
                payload = json.loads(body)
                return str(payload.get("error") or body)[:2000]
            except (ValueError, AttributeError):
                return body[:2000] or str(exc)
        except OSError:
            return str(exc)
    return str(exc)[:2000]


def _generation_settings(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ServiceError("تنظیمات تولید مدل معتبر نیست.")
    result = {}
    bounds = {
        "temperature": (0.0, 1.0, float),
        "top_p": (0.1, 1.0, float),
        "min_p": (0.0, 1.0, float),
        "repeat_penalty": (1.0, 1.5, float),
        "repeat_last_n": (0, 512, int),
        "top_k": (1, 100, int),
        "seed": (0, 2_147_483_647, int),
    }
    for key, raw in value.items():
        if key not in bounds:
            raise ServiceError(f"پارامتر تولید ناشناخته است: {key}")
        low, high, converter = bounds[key]
        try:
            converted = converter(raw)
        except (TypeError, ValueError, OverflowError):
            raise ServiceError(f"مقدار پارامتر {key} معتبر نیست.")
        if not low <= converted <= high:
            raise ServiceError(f"مقدار پارامتر {key} خارج از محدوده است.")
        result[key] = converted
    return result


class TaskService:
    def __init__(self, root_getter, ollama_post, capacity_getter, thinking_getter=lambda model: False, model_available=lambda model: False,
                 ollama_chat_cancellable=None, resources=None, auto_resume=lambda: False):
        self._root_getter = root_getter
        self._ollama_post = ollama_post
        self._capacity = capacity_getter
        self._thinking = thinking_getter
        self._model_available = model_available
        # optional ``f(body, stop_event) -> response dict`` that aborts the Ollama request as soon as stop_event is set
        self._chat_cancellable = ollama_chat_cancellable
        # optional object with prepare(model) -> list of released things and fit(model) -> {"gpu_percent", ...}
        self._resources = resources
        # a restart must not silently re-occupy the model: interrupted jobs resume automatically only when the user opted in
        self._auto_resume = auto_resume
        self._local = threading.local()                    # the running worker's stop event (set around each chunk)
        self._manager = None
        self._recovery_checked = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing
    @property
    def manager(self):
        should_recover = False
        with self._lock:
            if self._manager is None:
                self._manager = tasks.TaskManager(self._root_getter())
            if not self._recovery_checked:
                self._recovery_checked = True
                should_recover = True
            manager = self._manager
        # Do not hold the service lock while starting worker threads; start() accesses manager again.
        if should_recover:
            try:
                enabled = bool(self._auto_resume())
            except Exception:
                enabled = False
            if enabled:
                self._auto_resume_interrupted()
        return manager

    def _auto_resume_interrupted(self):
        """Resume only durable document jobs whose source and exact Ollama model still exist.

        Unsafe/unknown jobs remain interrupted and can still be resumed manually.
        Completed chunks are retained by TaskManager and are never sent again.
        """
        manager = self._manager
        if manager is None:
            return
        for item in manager.list(limit=100):
            if item.get("status") != tasks.INTERRUPTED:
                continue
            task = manager.get(item.get("id"))
            if not task or task.get("kind") not in KINDS:
                continue
            source = task.get("source_path")
            params = task.get("params") or {}
            model = str(params.get("model") or "").strip()
            if not source or not __import__("pathlib").Path(source).is_file() or not model:
                continue
            try:
                available = bool(self._model_available(model))
            except Exception:
                available = False
            if not available:
                continue
            try:
                self.start(task["id"])
                manager._event(task["id"], "info", "بازیابی خودکار پس از راه‌اندازی مجدد آغاز شد؛ بخش‌های تکمیل‌شده حفظ شدند.")
            except (tasks.TaskError, ServiceError, OSError) as exc:
                manager._event(task["id"], "warn", f"بازیابی خودکار انجام نشد؛ ادامه دستی ممکن است: {str(exc)[:240]}")

    def stop_all(self, timeout=5):
        if self._manager is not None:
            self._manager.stop_all(timeout)

    def generator(self, model, num_ctx, num_predict, temperature=0.2, generation_settings=None, strict_length=False):
        """``generate(messages) -> str`` bound to one model and window (no streaming: results are stored whole)."""
        thinking = bool(self._thinking(model))
        generation_settings = dict(generation_settings or {})
        allowed = {"temperature", "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n", "seed"}
        base_options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}
        base_options.update({key: value for key, value in generation_settings.items() if key in allowed})

        def generate_call(messages, recovery=False):
            options = dict(base_options)
            if recovery:
                options["temperature"] = min(0.5, max(0.25, float(options.get("temperature", 0.2)) + 0.05))
                options["num_predict"] = max(256, int(options["num_predict"] * 0.75))
                options["repeat_penalty"] = min(1.25, float(options.get("repeat_penalty", 1.1)) + 0.08)
                options["repeat_last_n"] = max(128, int(options.get("repeat_last_n", 64)))
                options["top_p"] = min(0.98, float(options.get("top_p", 0.9)))
                options["top_k"] = max(1, int(options.get("top_k", 40)))
            body = {"model": model, "stream": False, "keep_alive": runtime_guard.KEEP_ALIVE, "messages": messages,
                    "options": options}
            if "gpt-oss" in str(model).lower():
                body["think"] = "low"            # gpt-oss ignores think=false; "low" keeps its reasoning short so the answer fits num_predict
            elif thinking:
                body["think"] = False
            stop = getattr(self._local, "stop", None)
            try:
                data = self._chat(body, stop)
            except tasks.TaskCancelled:
                raise
            except Exception as exc:
                detail = _exception_detail(exc)
                if _is_repetition_error(detail):
                    context = getattr(self._local, "task_context", None)
                    task = getattr(context, "task", {}) or {}
                    chunk = getattr(context, "chunk", None)
                    log.exception(
                        "Ollama repetition abort: model=%s options=%s task_id=%s chunk=%s input_tokens=%s recovery=%s",
                        model, options, task.get("id", ""), getattr(chunk, "index", -1) + 1,
                        context_manager.estimate_messages_tokens(messages), recovery,
                    )
                    partial = getattr(exc, "partial", "")
                    raise tasks.RepetitionError(detail, partial=partial) from exc
                raise
            if strict_length and data.get("done_reason") == "length":
                raise tasks.OutputTruncated()
            return ((data.get("message") or {}).get("content") or "").strip()

        def generate(messages):
            return generate_call(messages)

        def recover(messages):
            return generate_call(messages, recovery=True)

        generate.recover = recover
        return generate

    def _chat(self, body, stop):
        """One Ollama chat call that gives up quickly when pause/cancel is requested (instead of finishing the chunk)."""
        if stop is None:
            return self._ollama_post("/api/chat", body, timeout=900)
        if stop.is_set():
            raise tasks.TaskCancelled()
        if self._chat_cancellable is not None:
            live = getattr(self._local, "live", None)
            try:
                if live is not None and self._cancellable_takes_on_token():
                    return self._chat_cancellable(body, stop, on_token=live)
                return self._chat_cancellable(body, stop)
            except InterruptedError:
                raise tasks.TaskCancelled()
        # Fallback: run the blocking call in a helper thread and stop waiting for it when asked to stop.
        box = {}

        def call():
            try:
                box["data"] = self._ollama_post("/api/chat", body, timeout=900)
            except BaseException as exc:                   # handed back to the worker thread below
                box["error"] = exc
        helper = threading.Thread(target=call, name="task-ollama-call", daemon=True)
        helper.start()
        while helper.is_alive():
            helper.join(0.25)
            if stop.is_set():
                raise tasks.TaskCancelled()
        if "error" in box:
            raise box["error"]
        return box.get("data") or {}

    def _cancellable_takes_on_token(self):
        try:
            return "on_token" in inspect.signature(self._chat_cancellable).parameters
        except (TypeError, ValueError):
            return False

    def window(self, params):
        """(num_ctx, num_predict, chunk_tokens) for the chosen model – the plan follows the model's real capacity."""
        capacity = self._capacity(params["model"])
        # default: use what the model really supports, but cap at 16k (VRAM and speed); the user can override
        requested = int(params.get("num_ctx") or min(capacity or 8192, DEFAULT_TASK_CTX))
        ctx = context_manager.effective_context(requested, capacity)
        predict = max(256, min(4096, int(ctx * 0.35)))
        expansion = 1.9 if params["kind"] == "translate" and params.get("target", "fa") in ("fa", "ar", "ru", "ja", "zh") else 1.0
        chunk = docengine.plan_chunk_tokens(ctx, predict, task=params["kind"], expansion=expansion)
        return ctx, predict, chunk

    # ------------------------------------------------------------------ one heavy job at a time
    def active_tasks(self):
        """Progress records of tasks that are running now (empty when nothing holds the local model)."""
        manager = self._manager
        if manager is None:
            return []
        return [p for p in (manager.progress(task_id) for task_id in manager.active_ids()) if p]

    def list(self, limit=50):
        items = self.manager.list(limit)
        for item in items:
            if not item.get("failed"):
                continue
            rows = self.manager.chunk_rows(item["id"], with_output=False)
            failures = []
            for row in rows:
                if row["status"] != "failed":
                    continue
                checkpoint = self.manager.recovery_checkpoint(item["id"], row["idx"], row.get("digest") or "")
                error = row.get("error") or ""
                if "RepetitionError" in error:
                    error = "تولید به‌دلیل تکرار غیرعادی متوقف شد؛ بخش قابل بازیابی با زیر‌بخش‌های کوچک‌تر است."
                failures.append({
                    "index": row["idx"] + 1, "status": row["status"], "attempts": row["attempts"],
                    "error": error, "recovery_completed": len(checkpoint.get("parts") or {}),
                    "recovery_partial": bool(checkpoint.get("last_partial")),
                })
            item["chunks"] = failures
        return items

    def ensure_idle(self, allow_task_id=None):
        """A local GPU cannot serve two heavy jobs at once; refuse the second one with a clear reason."""
        for task in self.active_tasks():
            if task["id"] != allow_task_id:
                label = "ترجمه" if task["kind"] == "translate" else "خلاصه‌سازی"
                raise ServiceError(
                    f"هم‌اکنون «{task['title']}» در حال {label} است ({task['done']} از {task['total']} بخش). "
                    "برای جلوگیری از فشار روی سخت‌افزار، تا پایان یا توقف آن کار جدیدی شروع نمی‌شود.", 409)

    # ------------------------------------------------------------------ create / control
    def create(self, name, data, options):
        self.manager                                       # make sure recovery ran before counting active jobs
        self.ensure_idle()
        kind = _clean(options.get("kind"), 20) or "translate"
        model = _clean(options.get("model"), 200)
        if kind not in KINDS:
            raise ServiceError("نوع کار باید translate یا summarize باشد.")
        if not model:
            raise ServiceError("مدل مشخص نشده است.")
        if not data or len(data) > MAX_TASK_UPLOAD:
            raise ServiceError("فایل خالی یا بزرگ‌تر از حد مجاز (۱۲۸ مگابایت) است.", 413)
        try:
            selection = pagespec.normalize(options.get("pages"))       # "only these PDF pages" (None = whole document)
        except pagespec.PageSpecError as exc:
            raise ServiceError(str(exc), 400)
        try:                                              # fail early, with a clear reason, before storing anything
            probe = docengine.load_document(bytes(data), name, page_spec=selection)
        except docengine.DocumentError as exc:
            raise ServiceError(str(exc), 415)
        if not probe.blocks:
            raise ServiceError("در این فایل متنی برای پردازش پیدا نشد.", 422)
        params = {"kind": kind, "model": model, "target": _clean(options.get("target"), 8) or "fa",
                  "style": _clean(options.get("style")), "num_ctx": int(options.get("num_ctx") or 0),
                  "workers": max(1, min(MAX_WORKERS, int(options.get("workers") or 1))),
                  "generation": _generation_settings(options.get("generation"))}
        if selection:
            params["pages"] = selection
        ctx, predict, chunk = self.window(params)
        params.update({"num_ctx": ctx, "num_predict": predict, "max_tokens": chunk})
        title = _clean(options.get("title"), 120) or name
        if selection:
            title = f"{title[:90]} (صفحه {pagespec.describe(selection)})"
        task_id = self.manager.create(kind, name, bytes(data), params, title=title)
        self.manager.prepare(task_id)                      # chunk now so the caller sees the plan immediately
        if options.get("autostart", True):
            self.start(task_id)
        return self.manager.progress(task_id)

    def processor(self, task):
        params = task["params"]
        generate = self.generator(params["model"], params["num_ctx"], params["num_predict"],
                                  generation_settings=params.get("generation"), strict_length=(task["kind"] == "translate"))
        if task["kind"] == "summarize":
            inner = pipeline.make_summarize_processor(generate, params.get("target", "fa"))
        else:
            inner = pipeline.make_translate_processor(generate, params.get("target", "fa"), params.get("style", ""))

        def processor(chunk, ctx):
            self._local.stop = ctx.stop                    # lets generate() notice pause/cancel mid-request
            self._local.task_context = ctx
            self._local.live = lambda n, thinking=False, _m=ctx.manager, _t=ctx.task["id"], _i=chunk.index: _m.live_tick(_t, _i, n, thinking)
            try:
                return inner(chunk, ctx)
            finally:
                self._local.stop = None
                self._local.task_context = None
                self._local.live = None
        return processor

    def start(self, task_id):
        task = self.manager.get(task_id)
        if task is None:
            raise ServiceError("کار پیدا نشد.", 404)
        self.ensure_idle(allow_task_id=task_id)
        model = str((task.get("params") or {}).get("model") or "")
        released = []
        if self._resources is not None and model:
            try:                                           # give the task model the whole GPU before it loads
                released = list(self._resources.prepare(model) or [])
            except Exception:
                log.warning("could not free resources before the task", exc_info=True)
        try:
            self.manager.start(task_id, self.processor(task), workers=task["params"].get("workers", 1))
        except tasks.TaskError as exc:
            raise ServiceError(str(exc), 409)
        if released:
            self.manager._event(task_id, "info", "برای سرعت بیشتر آزاد شد: " + "، ".join(released))
        if self._resources is not None and model:
            threading.Thread(target=self._watch_fit, args=(task_id, model), name="task-fit-watch", daemon=True).start()

    def _watch_fit(self, task_id, model):
        """While the task runs, keep the «how much of the model is on the GPU» figure fresh (cheap /api/ps call)."""
        time.sleep(3)
        warned = False
        while task_id in self.manager.active_ids():
            try:
                fit = self._resources.fit(model)
            except Exception:
                fit = None
            if fit:
                self.manager.set_fit(task_id, fit)
                if not warned and fit["gpu_percent"] < 85:
                    warned = True
                    self.manager._event(task_id, "warn", f"فقط {fit['gpu_percent']}٪ مدل روی GPU است و بقیه روی CPU/RAM؛ ترجمه کند می‌شود. مدل کوچک‌تر (مثلاً qwen3:8b) پیشنهاد می‌شود.")
            time.sleep(6)
        self.manager.set_fit(task_id, None)

    def detail(self, task_id):
        progress = self.manager.progress(task_id)
        if progress is None:
            raise ServiceError("کار پیدا نشد.", 404)
        rows = self.manager.chunk_rows(task_id, with_output=False)
        issues = []
        for row in rows:
            try:
                import json
                for issue in (json.loads(row["meta"] or "{}").get("issues") or []):
                    issues.append({"chunk": row["idx"] + 1, **issue})
            except ValueError:
                pass
        failed_chunks = []
        for row in rows:
            checkpoint = self.manager.recovery_checkpoint(task_id, row["idx"], row.get("digest") or "")
            failed_chunks.append({
                "index": row["idx"] + 1, "status": row["status"], "attempts": row["attempts"],
                "error": ("تولید به‌دلیل تکرار غیرعادی متوقف شد؛ می‌توانی همان بخش را با زیر‌بخش‌های کوچک‌تر دوباره اجرا کنی."
                          if "RepetitionError" in (row.get("error") or "") else row["error"]),
                "tokens": row["tokens_in"],
                "recovery_completed": len(checkpoint.get("parts") or {}),
                "recovery_partial": bool(checkpoint.get("last_partial")),
            })
        progress["chunks"] = failed_chunks
        last_error = progress.get("last_chunk_error")
        if last_error and "RepetitionError" in (last_error.get("error") or ""):
            last_error["technical_error"] = last_error["error"]
            last_error["error"] = "تولید به‌دلیل تکرار غیرعادی متوقف شد؛ بخش قابل بازیابی با زیر‌بخش‌های کوچک‌تر است."
        progress["issues"] = issues
        progress["events"] = self.manager.events(task_id, 30)
        progress["storage"] = self.manager.storage_stats(task_id)
        progress["glossary"] = self.manager.glossary(task_id)
        return progress

    def action(self, task_id, action, payload=None):
        payload = payload or {}
        if self.manager.get(task_id) is None:
            raise ServiceError("کار پیدا نشد.", 404)
        try:
            if action in ("start", "resume"):
                self.start(task_id)
            elif action == "pause":
                self.manager.pause(task_id)
            elif action == "cancel":
                self.manager.cancel(task_id)
            elif action == "retry":
                self.manager.retry_failed(task_id, payload.get("chunks"))
                self.start(task_id)
            elif action == "retry_smaller":
                retried = self.manager.retry_failed(task_id, payload.get("chunks"))
                if not retried:
                    raise ServiceError("بخش ناموفقی برای بازیابی پیدا نشد.", 409)
                task = self.manager.get(task_id)
                params = dict(task.get("params") or {})
                params["recovery_smaller"] = True
                self.manager._set(task_id, params=params)
                self.manager._event(task_id, "info", "بخش‌های ناموفق با تقسیم‌بندی کوچک‌تر برای بازیابی آماده شدند.")
                self.start(task_id)
            elif action == "regenerate":
                indexes = [int(i) - 1 for i in (payload.get("chunks") or [])]
                if not indexes:
                    raise ServiceError("شمارهٔ بخش‌ها مشخص نشده است.")
                self.manager.regenerate(task_id, indexes)
                self.start(task_id)
            elif action == "delete":
                self.manager.delete(task_id)
                return {"deleted": True}
            elif action == "glossary":
                pairs = payload.get("terms")
                if not isinstance(pairs, dict) or not pairs:
                    raise ServiceError("واژه‌نامه معتبر نیست.")
                self.manager.add_glossary(task_id, {_clean(k, 60): _clean(v, 60) for k, v in pairs.items()}, locked=True)
            else:
                raise ServiceError("عملیات ناشناخته است.", 404)
        except tasks.TaskError as exc:
            raise ServiceError(str(exc), 409)
        return self.detail(task_id)

    def download(self, task_id, fmt="docx", allow_partial=False):
        task = self.manager.get(task_id)
        if task is None:
            raise ServiceError("کار پیدا نشد.", 404)
        fmt = (fmt or "docx").lower()
        if fmt not in docwriter.FORMATS:
            raise ServiceError("قالب باید txt، md، docx یا pdf باشد.")
        try:
            data, report = pipeline.assemble(self.manager, task_id, fmt, allow_partial=allow_partial)
        except tasks.TaskError as exc:
            raise ServiceError(str(exc), 409)
        except docwriter.WriterUnavailable as exc:
            raise ServiceError(str(exc), 501)
        stem = re.sub(r"[^\w\-.]+", "_", task["title"].rsplit(".", 1)[0], flags=re.U).strip("_") or "output"
        return data, docwriter.FORMATS[fmt][1], f"{stem}.{fmt}", report
