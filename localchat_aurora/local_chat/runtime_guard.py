"""Runtime guard: shows WHY the model memory / context is occupied and frees it automatically when idle.

Nothing here calls a language model; it only reads what Ollama reports (/api/ps) and decides.

* ``summarize_loaded`` – every model Ollama holds, with its real window (``context_length``), the part that sits
  in VRAM and when Ollama itself would drop it.
* ``diagnose`` – plain Persian findings (what holds the memory, and what to do about it).
* ``IdleGuardian`` – a small daemon: when no chat reply and no long task has run for N minutes, every resident
  model is unloaded (``keep_alive: 0``). This is what makes «restart / new chat does not free the context»
  stop being a problem: nothing stays resident longer than needed, and nothing restarts by itself
  (see ``settings.runtime_settings`` → ``auto_resume_tasks``).
"""
import threading
import time
from datetime import datetime, timezone

# One keep-alive for every part of the app (chat, long tasks, helpers). Different values per call site used to
# keep several models resident for 5 / 10 / 20 minutes each.
KEEP_ALIVE = "5m"
DEFAULT_IDLE_UNLOAD_MINUTES = 5
MAX_IDLE_UNLOAD_MINUTES = 240
BIG_WINDOW = 12288          # a window above this makes the KV cache a large part of the memory
_GB = 1024 ** 3


def _parse_time(value):
    """Ollama sends RFC 3339 with nanoseconds and a zone; datetime wants microseconds at most."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    if "." in text:
        head, rest = text.split(".", 1)
        digits = ""
        while rest and rest[0].isdigit():
            digits, rest = digits + rest[0], rest[1:]
        text = f"{head}.{digits[:6].ljust(6, '0')}{rest}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def summarize_loaded(items, now=None):
    """Normalise /api/ps entries for the UI."""
    now = now or datetime.now(timezone.utc)
    models = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        size, vram = int(item.get("size") or 0), int(item.get("size_vram") or 0)
        expires = _parse_time(item.get("expires_at"))
        left = max(0, int((expires - now).total_seconds())) if expires else None
        models.append({
            "name": name,
            "size_gb": round(size / _GB, 1),
            "vram_gb": round(vram / _GB, 1),
            "gpu_percent": max(0, min(100, round(100 * vram / size))) if size > 0 else None,
            "context": item.get("context_length") if isinstance(item.get("context_length"), int) else None,
            "expires_in": left,
        })
    return models


def _minutes(seconds):
    return max(1, round(seconds / 60))


def diagnose(models, *, task_reason="", chat_busy=False, auto_resume=False, idle_minutes=DEFAULT_IDLE_UNLOAD_MINUTES,
             voice_running=False):
    """Findings as ``[{"level": "info"|"warn", "text": ...}]`` — each says what holds memory and what to do."""
    findings = []
    if task_reason:
        findings.append({"level": "info", "text": "یک کار طولانی (ترجمه/خلاصه‌سازی) مدل را نگه داشته است؛ تا پایان یا توقف موقت آن "
                                                    "حافظه آزاد نمی‌شود."})
    if chat_busy:
        findings.append({"level": "info", "text": "مدل هم‌اکنون در حال پاسخ به یک پیام است."})
    if len(models) > 1:
        names = "، ".join(m["name"] for m in models)
        findings.append({"level": "warn", "text": f"{len(models)} مدل هم‌زمان در حافظه‌اند ({names}) و هرکدام پنجرهٔ کانتکست جدا دارد. "
                                                    "«آزادسازی حافظه» همه را خالی می‌کند."})
    for model in models:
        if model["gpu_percent"] is not None and model["gpu_percent"] < 100:
            findings.append({"level": "warn", "text": f"مدل «{model['name']}» فقط {model['gpu_percent']}٪ روی GPU است و بقیه روی RAM/CPU "
                                                        "اجرا می‌شود (کند). پنجرهٔ کوچک‌تر یا مدل کوچک‌تر کمک می‌کند."})
        if model["context"] and model["context"] > BIG_WINDOW:
            findings.append({"level": "warn", "text": f"پنجرهٔ «{model['name']}» {model['context']} توکن است؛ بخش بزرگی از حافظه "
                                                        "صرف KV-cache می‌شود. برای ترجمه و چت معمولی ۸٬۱۹۲ کافی است."})
    idle = not task_reason and not chat_busy
    if models and idle:
        waits = [m["expires_in"] for m in models if m["expires_in"] is not None]
        soon = _minutes(min(waits)) if waits else None
        if idle_minutes:
            findings.append({"level": "info", "text": f"کاری در جریان نیست؛ مدل‌ها بعد از {idle_minutes} دقیقه بی‌کاری خودکار آزاد می‌شوند"
                                                        + (f" (Ollama خودش حدود {soon} دقیقهٔ دیگر)." if soon else ".")})
        else:
            findings.append({"level": "warn", "text": "آزادسازی خودکار هنگام بی‌کاری خاموش است؛ مدل‌ها تا پایان keep_alive "
                                                        "(حدود ۵ دقیقه) در حافظه می‌مانند."})
    if voice_running:
        findings.append({"level": "info", "text": "موتور صدا (TTS/STT) هم VRAM نگه داشته است."})
    if auto_resume:
        findings.append({"level": "warn", "text": "ازسرگیری خودکار کارهای قطع‌شده روشن است: بعد از هر راه‌اندازی مجدد، ترجمهٔ نیمه‌تمام "
                                                    "دوباره مدل را بار می‌کند."})
    if not models and not findings:
        findings.append({"level": "info", "text": "هیچ مدلی در حافظه نیست؛ همه‌چیز آزاد است."})
    return findings


class IdleGuardian:
    """Unloads every resident model after ``minutes`` without a chat reply or a long task."""

    def __init__(self, list_loaded, unload, is_busy, minutes_getter, interval=20, clock=time.monotonic):
        self._list_loaded = list_loaded
        self._unload = unload
        self._is_busy = is_busy
        self._minutes = minutes_getter
        self._interval = interval
        self._clock = clock
        self._idle_since = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_release = None            # {"at": epoch seconds, "models": [...], "reason": "idle"|"manual"}

    def tick(self):
        """One check; returns the names released by this call."""
        try:
            minutes = int(self._minutes() or 0)
        except (TypeError, ValueError):
            minutes = 0
        now = self._clock()
        if self._is_busy():
            self._idle_since = None
            return []
        if self._idle_since is None:
            self._idle_since = now
        if minutes <= 0 or now - self._idle_since < minutes * 60:
            return []
        released = self.release_all("idle")
        self._idle_since = now              # re-arm; the next look finds nothing loaded and does nothing
        return released

    def release_all(self, reason="manual"):
        released = []
        for item in self._list_loaded() or ():
            name = str(item.get("name") or item.get("model") or "") if isinstance(item, dict) else ""
            if not name:
                continue
            try:
                if self._unload(name):
                    released.append(name)
            except Exception:  # noqa: BLE001 - a model that cannot be unloaded must not stop the others
                continue
        if released:
            self.last_release = {"at": time.time(), "models": released, "reason": reason}
        return released

    def ensure_started(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="runtime-idle-guardian", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the guard must never die because Ollama was down for a moment
                continue
