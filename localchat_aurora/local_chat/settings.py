"""Persistent app settings for LocalChat (Ollama models directory, web-search engine)."""
import json
import os
import re
import threading
from pathlib import Path

_LOCK = threading.Lock()


def app_data_dir():
    # Portable builds set LOCALCHAT_DATA_DIR to a folder beside the app.
    # Installed builds leave it unset and use the standard per-user AppData location.
    portable = os.environ.get("LOCALCHAT_DATA_DIR", "").strip()
    if portable:
        return Path(portable).expanduser().resolve()
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "LocalChat"


def settings_path():
    return app_data_dir() / "settings.json"


def load_settings():
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_settings(data):
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def default_models_dir():
    return Path.home() / ".ollama" / "models"


def configured_models_dir():
    """Directory the user chose in the app, or '' when none was chosen."""
    value = load_settings().get("models_dir")
    return value.strip() if isinstance(value, str) else ""


def effective_models_dir():
    """Directory Ollama should use: app setting, else OLLAMA_MODELS, else default."""
    chosen = configured_models_dir()
    if chosen:
        return Path(chosen)
    env_value = os.environ.get("OLLAMA_MODELS", "").strip()
    return Path(env_value) if env_value else default_models_dir()


def inspect_models_dir(path):
    """Describe a candidate models directory without modifying it."""
    folder = Path(path)
    info = {
        "path": str(folder),
        "exists": folder.is_dir(),
        "looks_like_ollama": False,
        "model_count": 0,
    }
    if not info["exists"]:
        return info
    manifests = folder / "manifests"
    info["looks_like_ollama"] = manifests.is_dir() and (folder / "blobs").is_dir()
    if manifests.is_dir():
        try:
            info["model_count"] = sum(1 for p in manifests.rglob("*") if p.is_file())
        except OSError:
            pass
    return info


def set_models_dir(raw_path):
    """Validate and store the models directory. Empty string resets to default.

    Raises ValueError with a Persian message for invalid input.
    """
    if not isinstance(raw_path, str):
        raise ValueError("مسیر معتبر نیست.")
    value = raw_path.strip().strip('"')
    with _LOCK:
        data = load_settings()
        if not value:
            data.pop("models_dir", None)
            _save_settings(data)
            return inspect_models_dir(effective_models_dir())
        if len(value) > 500 or "\x00" in value:
            raise ValueError("مسیر معتبر نیست.")
        folder = Path(os.path.expandvars(value)).expanduser()
        if not folder.is_absolute():
            raise ValueError("مسیر باید کامل باشد، مثلاً D:\\OllamaModels")
        if not folder.is_dir():
            raise ValueError("این پوشه وجود ندارد. ابتدا پوشه را بسازید یا مسیر درست را وارد کنید.")
        data["models_dir"] = str(folder)
        _save_settings(data)
        return inspect_models_dir(folder)


def default_output_dir():
    return Path.home() / "Documents" / "LocalChat" / "Outputs"


def effective_output_dir():
    """Folder where finished translations / summaries are saved: app setting, else Documents\\LocalChat\\Outputs."""
    value = load_settings().get("output_dir")
    if isinstance(value, str) and value.strip():
        return Path(os.path.expandvars(value.strip())).expanduser()
    return default_output_dir()


def set_output_dir(raw_path):
    """Store the output folder (created when missing). Empty string resets to the default. ValueError carries a Persian message."""
    if not isinstance(raw_path, str):
        raise ValueError("مسیر معتبر نیست.")
    value = raw_path.strip().strip('"')
    with _LOCK:
        data = load_settings()
        if not value:
            data.pop("output_dir", None)
            _save_settings(data)
            return effective_output_dir()
        if len(value) > 500 or "\x00" in value:
            raise ValueError("مسیر معتبر نیست.")
        folder = Path(os.path.expandvars(value)).expanduser()
        if not folder.is_absolute():
            raise ValueError("مسیر باید کامل باشد، مثلاً D:\\Translations")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".localchat_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            raise ValueError("در این پوشه نمی‌توان فایل ذخیره کرد؛ مسیر یا دسترسی را بررسی کن.")
        data["output_dir"] = str(folder)
        _save_settings(data)
        return folder


def ollama_environment():
    """Environment for launching `ollama serve` with the chosen models directory."""
    env = os.environ.copy()
    chosen = configured_models_dir()
    if chosen:
        env["OLLAMA_MODELS"] = chosen
    return env


# --- web search -------------------------------------------------------------------

SEARCH_ENGINES = ("auto", "google", "brave", "duckduckgo", "searxng")
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-.]{8,200}$")
_CX_PATTERN = re.compile(r"^[A-Za-z0-9:_\-.]{3,120}$")
SEARCH_KEY_FIELDS = {
    "serper_api_key": _KEY_PATTERN,
    "brave_api_key": _KEY_PATTERN,
    "google_api_key": _KEY_PATTERN,
    "google_cx": _CX_PATTERN,
}


def search_settings():
    """Full search settings including secrets (server-side use only)."""
    data = load_settings().get("search", {})
    data = data if isinstance(data, dict) else {}
    engine = data.get("engine")
    result = {"engine": engine if engine in SEARCH_ENGINES else "auto"}
    searxng_url = os.environ.get("LOCALCHAT_SEARXNG_URL", data.get("searxng_url", ""))
    result["searxng_url"] = searxng_url.strip().rstrip("/") if isinstance(searxng_url, str) else ""
    for field in SEARCH_KEY_FIELDS:
        value = data.get(field)
        result[field] = value.strip() if isinstance(value, str) else ""
    return result


def _hint(value):
    return ("\u2022\u2022\u2022\u2022" + value[-4:]) if value else ""


def search_settings_public():
    """What the browser may see: never the secrets themselves."""
    full = search_settings()
    return {
        "engine": full["engine"],
        "has_serper_key": bool(full["serper_api_key"]),
        "has_brave_key": bool(full["brave_api_key"]),
        "brave_hint": _hint(full["brave_api_key"]),
        "serper_hint": _hint(full["serper_api_key"]),
        "has_google_cse": bool(full["google_api_key"] and full["google_cx"]),
        "google_api_hint": _hint(full["google_api_key"]),
        "google_cx": full["google_cx"],
        "searxng_url": full["searxng_url"],
        "has_searxng": bool(full["searxng_url"]),
    }


def set_search_settings(update):
    """Apply a partial update. Missing field = keep, empty string = clear."""
    if not isinstance(update, dict):
        raise ValueError("تنظیمات جست‌وجو معتبر نیست.")
    with _LOCK:
        data = load_settings()
        search = data.get("search", {})
        search = dict(search) if isinstance(search, dict) else {}
        if "engine" in update:
            if update["engine"] not in SEARCH_ENGINES:
                raise ValueError("موتور جست‌وجو معتبر نیست.")
            search["engine"] = update["engine"]
        if "searxng_url" in update:
            value = update["searxng_url"]
            if not isinstance(value, str):
                raise ValueError("نشانی SearXNG معتبر نیست.")
            value = value.strip().rstrip("/")
            if not value:
                search.pop("searxng_url", None)
            else:
                from urllib.parse import urlsplit
                parts = urlsplit(value)
                if parts.scheme != "https" or not parts.hostname or parts.path not in ("", "/") or parts.username or parts.password or parts.query or parts.fragment:
                    raise ValueError("نشانی SearXNG باید HTTPS و بدون مسیر، پارامتر یا اطلاعات ورود باشد.")
                search["searxng_url"] = value
        for field, pattern in SEARCH_KEY_FIELDS.items():
            if field not in update:
                continue
            value = update[field]
            if not isinstance(value, str):
                raise ValueError("مقدار کلید معتبر نیست.")
            value = value.strip()
            if not value:
                search.pop(field, None)
            elif not pattern.fullmatch(value):
                raise ValueError("قالب کلید یا شناسهٔ موتور جست‌وجو معتبر نیست.")
            else:
                search[field] = value
        data["search"] = search
        _save_settings(data)
    return search_settings_public()


# --- speech synthesis settings ---------------------------------------------------
TTS_PROVIDERS = ("edge_tts", "local_parsvoice")
TTS_VOICES = {"edge_tts": ("fa-IR-DilaraNeural", "fa-IR-FaridNeural"), "local_parsvoice": ("reference",)}

def tts_settings():
    data = load_settings().get("tts", {})
    data = data if isinstance(data, dict) else {}
    provider = data.get("provider", "edge_tts")
    if provider not in TTS_PROVIDERS: provider = "edge_tts"
    voice = data.get("voice", "fa-IR-DilaraNeural")
    if voice not in TTS_VOICES[provider]: voice = TTS_VOICES[provider][0]
    rate = data.get("rate", "+0%")
    if not isinstance(rate, str) or not re.fullmatch(r"[+-](?:[0-5]?[0-9]|60)%", rate): rate = "+0%"
    model_dir = data.get("model_dir", "")
    python_path = data.get("python_path", "")
    return {"provider": provider, "voice": voice, "rate": rate, "model_dir": model_dir if isinstance(model_dir, str) else "", "fallback_local": bool(data.get("fallback_local", False)),
            "python_path": python_path if isinstance(python_path, str) else ""}

def tts_settings_public():
    return tts_settings()

def set_tts_settings(update):
    if not isinstance(update, dict): raise ValueError("تنظیمات صوتی معتبر نیست.")
    current = tts_settings()
    provider = update.get("provider", current["provider"])
    if provider not in TTS_PROVIDERS: raise ValueError("موتور تولید صدا معتبر نیست.")
    voice = update.get("voice", current["voice"] if current["provider"] == provider else TTS_VOICES[provider][0])
    if voice not in TTS_VOICES[provider]: raise ValueError("صدای انتخاب‌شده برای این موتور معتبر نیست.")
    rate = update.get("rate", current["rate"])
    if not isinstance(rate, str) or not re.fullmatch(r"[+-](?:[0-5]?[0-9]|60)%", rate): raise ValueError("سرعت گفتار باید بین منفی ۶۰ تا مثبت ۶۰ درصد باشد.")
    model_dir = update.get("model_dir", current["model_dir"])
    if not isinstance(model_dir, str) or len(model_dir) > 800 or "\x00" in model_dir: raise ValueError("مسیر مدل معتبر نیست.")
    if model_dir.strip() and not Path(os.path.expandvars(model_dir)).expanduser().is_dir(): raise ValueError("پوشهٔ مدل ParsVoice پیدا نشد.")
    python_path = update.get("python_path", current["python_path"])
    if not isinstance(python_path, str) or len(python_path) > 800 or "\x00" in python_path: raise ValueError("مسیر python.exe معتبر نیست.")
    python_path = python_path.strip().strip('"')
    if python_path:
        candidate = Path(os.path.expandvars(python_path)).expanduser()
        if candidate.is_dir():  # accept the venv folder itself as well as python.exe
            for sub in (("Scripts", "python.exe"), ("bin", "python"), ("python.exe",)):
                if candidate.joinpath(*sub).is_file(): candidate = candidate.joinpath(*sub); break
        if not candidate.is_file(): raise ValueError("فایل python.exe محیط مجازی پیدا نشد.")
        python_path = str(candidate)
    result = {"provider": provider, "voice": voice, "rate": rate, "model_dir": model_dir.strip(), "fallback_local": bool(update.get("fallback_local", current["fallback_local"])),
              "python_path": python_path}
    with _LOCK:
        data = load_settings(); data["tts"] = result; _save_settings(data)
    return result


# --- Forex analysis settings (MetaTrader 5) -------------------------------------
FOREX_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")
_SUFFIX_PATTERN = re.compile(r"^[A-Za-z0-9._#\-+]{0,12}$")
_TERMINAL_NAMES = {"terminal64.exe"}


def forex_settings():
    data = load_settings().get("forex", {})
    data = data if isinstance(data, dict) else {}
    suffix = data.get("symbol_suffix", "")
    timeframe = data.get("default_timeframe", "M15")
    terminal = data.get("terminal_path", "")
    return {
        "enabled": data.get("enabled") if isinstance(data.get("enabled"), bool) else True,
        "terminal_path": terminal.strip() if isinstance(terminal, str) else "",
        "symbol_suffix": suffix.strip() if isinstance(suffix, str) and _SUFFIX_PATTERN.fullmatch(suffix.strip()) else "",
        "default_timeframe": timeframe if timeframe in FOREX_TIMEFRAMES else "M15",
        "news_enabled": data.get("news_enabled") if isinstance(data.get("news_enabled"), bool) else True,
        "news_url": data.get("news_url", "") if isinstance(data.get("news_url", ""), str) else "",
        "use_vision": True,
        "vision_model": "",
        "text_model": "",
    }


def set_forex_settings(update):
    if not isinstance(update, dict):
        raise ValueError("تنظیمات فارکس معتبر نیست.")
    with _LOCK:
        data = load_settings()
        forex = data.get("forex", {})
        forex = dict(forex) if isinstance(forex, dict) else {}
        if "enabled" in update:
            if not isinstance(update["enabled"], bool):
                raise ValueError("مقدار فعال/غیرفعال معتبر نیست.")
            forex["enabled"] = update["enabled"]
        if "symbol_suffix" in update:
            value = update["symbol_suffix"]
            if not isinstance(value, str) or not _SUFFIX_PATTERN.fullmatch(value.strip()):
                raise ValueError("پسوند نماد معتبر نیست.")
            forex["symbol_suffix"] = value.strip()
        if "default_timeframe" in update:
            if update["default_timeframe"] not in FOREX_TIMEFRAMES:
                raise ValueError("تایم‌فریم پیش‌فرض معتبر نیست.")
            forex["default_timeframe"] = update["default_timeframe"]
        if "terminal_path" in update:
            value = update["terminal_path"]
            if not isinstance(value, str):
                raise ValueError("مسیر ترمینال معتبر نیست.")
            value = value.strip().strip('"')
            if value:
                path = Path(os.path.expandvars(value)).expanduser()
                if not path.is_absolute() or path.name.lower() != "terminal64.exe" or not path.is_file():
                    raise ValueError("مسیر باید به فایل معتبر terminal64.exe اشاره کند.")
                forex["terminal_path"] = str(path)
            else:
                forex.pop("terminal_path", None)
        if "news_enabled" in update:
            if not isinstance(update["news_enabled"], bool):
                raise ValueError("وضعیت تقویم خبری معتبر نیست.")
            forex["news_enabled"] = update["news_enabled"]
        data["forex"] = forex
        _save_settings(data)
    return forex_settings()


# ---------------------------------------------------------------------------------- runtime (memory / context)
RUNTIME_DEFAULTS = {"auto_resume_tasks": False, "idle_unload_minutes": 5, "max_context": 32768}
# 0 = «auto»: no user ceiling, only the model's own declared limit and the hardware plan apply
CONTEXT_CHOICES = (0, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144)
MAX_IDLE_UNLOAD_MINUTES = 240


def runtime_settings():
    """Memory behaviour: automatic resume of interrupted tasks (off by default) and idle unload minutes (0 = off)."""
    data = load_settings().get("runtime", {})
    data = data if isinstance(data, dict) else {}
    auto = data.get("auto_resume_tasks")
    minutes = data.get("idle_unload_minutes")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 0 <= minutes <= MAX_IDLE_UNLOAD_MINUTES:
        minutes = RUNTIME_DEFAULTS["idle_unload_minutes"]
    ceiling = data.get("max_context")
    if isinstance(ceiling, bool) or ceiling not in CONTEXT_CHOICES:
        ceiling = RUNTIME_DEFAULTS["max_context"]
    return {"auto_resume_tasks": auto if isinstance(auto, bool) else RUNTIME_DEFAULTS["auto_resume_tasks"],
            "idle_unload_minutes": minutes, "max_context": ceiling}


def set_runtime_settings(update):
    if not isinstance(update, dict):
        raise ValueError("تنظیمات حافظه معتبر نیست.")
    with _LOCK:
        data = load_settings()
        runtime = data.get("runtime", {})
        runtime = dict(runtime) if isinstance(runtime, dict) else {}
        if "auto_resume_tasks" in update:
            if not isinstance(update["auto_resume_tasks"], bool):
                raise ValueError("مقدار ازسرگیری خودکار معتبر نیست.")
            runtime["auto_resume_tasks"] = update["auto_resume_tasks"]
        if "idle_unload_minutes" in update:
            value = update["idle_unload_minutes"]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_IDLE_UNLOAD_MINUTES:
                raise ValueError(f"زمان آزادسازی خودکار باید بین ۰ و {MAX_IDLE_UNLOAD_MINUTES} دقیقه باشد (۰ = خاموش).")
            runtime["idle_unload_minutes"] = value
        if "max_context" in update:
            value = update["max_context"]
            if isinstance(value, bool) or value not in CONTEXT_CHOICES:
                raise ValueError("سقف پنجرهٔ کانتکست باید یکی از " + "، ".join(str(c) for c in CONTEXT_CHOICES if c) + " یا ۰ (خودکار) باشد.")
            runtime["max_context"] = value
        data["runtime"] = runtime
        _save_settings(data)
    return runtime_settings()


def clock_state():
    """Saved correction of the internal clock: {offset, synced_at, source} (empty before the first sync)."""
    data = load_settings().get("clock", {})
    if not isinstance(data, dict):
        return {}
    offset, synced = data.get("offset"), data.get("synced_at")
    ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    return {"offset": float(offset) if ok(offset) else 0.0, "synced_at": float(synced) if ok(synced) else None,
            "source": str(data.get("source") or "")[:80]}


def set_clock_state(state):
    with _LOCK:
        data = load_settings()
        data["clock"] = {"offset": float(state.get("offset", 0.0)), "synced_at": float(state["synced_at"]),
                         "source": str(state.get("source") or "")[:80]}
        _save_settings(data)
