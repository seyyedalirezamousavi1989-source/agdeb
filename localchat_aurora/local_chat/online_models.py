"""Online (cloud) models, starting with Claude through the Anthropic Messages API and the user's own API key.

Design
* An online model is addressed as ``online:<provider>/<model id>`` (e.g. ``online:anthropic/claude-sonnet-5-5``) so it
  fits the existing model-name validation and can sit in the same model selector as Ollama models.
* The key lives in settings.json. On Windows it is protected with DPAPI (readable only by the same Windows user);
  elsewhere it is stored as-is with owner-only file permissions. The key is NEVER sent back to the browser: the UI only
  learns «a key is set» and its last four characters.
* ``open_stream`` translates the Anthropic streaming (SSE) format into the NDJSON lines the app already consumes from
  Ollama, so the chat loop, token accounting and history work unchanged.
* Everything that is sent leaves this computer. The UI says so; nothing is sent unless an online model is chosen.
"""
import base64
import ctypes
import json
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from . import settings as app_settings
except ImportError:                      # pragma: no cover - script-style launch
    import settings as app_settings

PREFIX = "online:"
API_VERSION = "2023-06-01"
PROVIDERS = {
    "anthropic": {"label": "Claude (Anthropic)", "base": "https://api.anthropic.com", "key_hint": "sk-ant-…",
                  "console": "https://console.anthropic.com/settings/keys", "style": "anthropic",
                  "ctx": 16384, "predict": 4096},
    # Free tiers: the three below are OpenAI-compatible and have per-minute / per-day token caps, so each gets a SMALL
    # working window and answer limit (the whole history is re-sent every turn and max_tokens counts against the cap).
    "gemini": {"label": "Gemini (Google AI Studio، رایگان)", "base": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_hint": "AIza…", "console": "https://aistudio.google.com/apikey", "style": "openai",
               "ctx": 24576, "predict": 4096, "usage": False},
    "groq": {"label": "Groq (رایگان)", "base": "https://api.groq.com/openai/v1", "key_hint": "gsk_…",
             "console": "https://console.groq.com/keys", "style": "openai", "ctx": 8192, "predict": 1536, "usage": True},
    "openrouter": {"label": "OpenRouter (فقط مدل‌های رایگان :free)", "base": "https://openrouter.ai/api/v1",
                   "key_hint": "sk-or-v1-…", "console": "https://openrouter.ai/keys", "style": "openai",
                   "ctx": 16384, "predict": 2048, "usage": True},
}
# Used only when the key is valid but the model list cannot be fetched; clearly marked as a default list in the UI.
FALLBACK_MODELS = {
    "anthropic": [("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("claude-opus-5-5", "Claude Opus 5.5"),
                  ("claude-haiku-5-5", "Claude Haiku 5.5")],
    "gemini": [("gemini-2.5-flash", "Gemini 2.5 Flash"), ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite"),
               ("gemini-2.5-pro", "Gemini 2.5 Pro")],
    "groq": [("llama-3.3-70b-versatile", "Llama 3.3 70B"), ("llama-3.1-8b-instant", "Llama 3.1 8B Instant"),
             ("openai/gpt-oss-20b", "GPT-OSS 20B")],
    "openrouter": [("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (free)"),
                   ("google/gemma-3-27b-it:free", "Gemma 3 27B (free)")],
}
_SKIP_WORDS = ("embed", "whisper", "tts", "guard", "moderation", "imagen", "veo", "aqa", "image-generation", "transcribe",
               "playai", "orpheus", "robotics", "live", "audio")
_VISION_HINTS = ("vision", "-vl", "vl-", "llama-4", "gemma-3", "gemini", "gpt-4o", "pixtral", "llava", "claude")
DEFAULT_MAX_TOKENS = 4096
_LIST_TTL = 600.0
_LOCK = threading.Lock()
_CACHE = {}


class OnlineError(ValueError):
    """A failure talking to an online provider, with a user-readable (Persian) message."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message, self.status = message, status


# ---------------------------------------------------------------------------------------------- names
def is_online(name):
    return isinstance(name, str) and name.startswith(PREFIX)


def make_name(provider, model_id):
    return f"{PREFIX}{provider}/{model_id}"


def split_name(name):
    """``online:anthropic/claude-x`` -> ("anthropic", "claude-x"); raises OnlineError for anything else."""
    if is_online(name):
        provider, _, model_id = name[len(PREFIX):].partition("/")
        if provider in PROVIDERS and model_id:
            return provider, model_id
    raise OnlineError("نام مدل آنلاین معتبر نیست.")


# ---------------------------------------------------------------------------------------------- key storage
class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(data, protect):
    blob_in = _Blob(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_char)))
    blob_out = _Blob()
    func = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    # both functions share the layout (data in, description, entropy, reserved, prompt, flags, data out)
    if not func(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise OSError("DPAPI failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def protect(text):
    raw = text.encode("utf-8")
    if os.name == "nt":
        try:
            return "dpapi:" + base64.b64encode(_dpapi(raw, True)).decode("ascii")
        except (OSError, AttributeError):
            pass
    return "plain:" + base64.b64encode(raw).decode("ascii")


def unprotect(stored):
    kind, _, body = str(stored).partition(":")
    try:
        raw = base64.b64decode(body)
        if kind == "dpapi":
            raw = _dpapi(raw, False)
        elif kind != "plain":
            return ""
        return raw.decode("utf-8")
    except (ValueError, OSError, AttributeError):
        return ""


def _online_section(settings):
    section = settings.get("online")
    return section if isinstance(section, dict) else {}


def get_key(provider):
    entry = _online_section(app_settings.load_settings()).get(provider)
    return unprotect(entry.get("key", "")) if isinstance(entry, dict) else ""


def set_key(provider, key):
    if provider not in PROVIDERS:
        raise OnlineError("ارائه‌دهندهٔ ناشناخته.")
    key = (key or "").strip()
    if not 20 <= len(key) <= 400 or not key.isascii() or any(c.isspace() for c in key):
        raise OnlineError("کلید API معتبر نیست (باید یک رشتهٔ ASCII بدون فاصله باشد).")
    with app_settings._LOCK:
        data = app_settings.load_settings()
        section = dict(_online_section(data))
        section[provider] = {"key": protect(key), "saved": int(time.time())}
        data["online"] = section
        app_settings._save_settings(data)
        _restrict_permissions()
    with _LOCK:
        _CACHE.pop(provider, None)


def delete_key(provider):
    with app_settings._LOCK:
        data = app_settings.load_settings()
        section = dict(_online_section(data))
        section.pop(provider, None)
        data["online"] = section
        app_settings._save_settings(data)
    with _LOCK:
        _CACHE.pop(provider, None)


def _restrict_permissions():
    try:
        if os.name != "nt":
            os.chmod(app_settings.settings_path(), 0o600)
    except OSError:
        pass


def key_status(provider):
    """What the UI may know: whether a key exists and its last four characters (never the key)."""
    key = get_key(provider)
    return {"has_key": bool(key), "tail": key[-4:] if key else ""}


# ---------------------------------------------------------------------------------------------- HTTP helpers
def _headers(provider, key):
    # a browser-like User-Agent: some providers' CDNs (Groq) reject urllib's default one with 403
    base = {"content-type": "application/json", "user-agent": "LocalChat/2 (+local)"}
    if PROVIDERS[provider]["style"] == "anthropic":
        return {**base, "x-api-key": key, "anthropic-version": API_VERSION}
    return {**base, "authorization": f"Bearer {key}"}


def _error_detail(raw):
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return ""
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(error or "")


def _explain_http_error(exc):
    try:
        detail = _error_detail(exc.read())
    except OSError:
        detail = ""
    table = {400: "درخواست توسط سرویس آنلاین پذیرفته نشد.",
             401: "کلید API نامعتبر یا منقضی است.", 403: "این کلید اجازهٔ استفاده از این مدل را ندارد.",
             404: "این مدل برای کلید شما پیدا نشد.", 413: "درخواست برای مدل آنلاین بیش از حد بزرگ است (سقف توکن این سرویس رایگان)؛ گفتگوی جدید شروع کن یا پیوست را کوچک‌تر کن.",
             429: "سقف نرخ یا اعتبار این کلید پر شده است (در سرویس‌های رایگان سقف توکن در دقیقه/روز دارد)؛ کمی بعد دوباره تلاش کن.",
             529: "سرویس آنلاین موقتاً شلوغ است؛ دوباره تلاش کن."}
    base = table.get(exc.code, f"سرویس آنلاین درخواست را نپذیرفت (HTTP {exc.code}).")
    return OnlineError(f"{base} {detail[:300]}".strip(), exc.code)


def _network_error(exc):
    return OnlineError(f"ارتباط با سرویس آنلاین برقرار نشد (اینترنت یا فایروال را بررسی کن): {exc}")


# ---------------------------------------------------------------------------------------------- model list
def _parse_models(provider, items):
    """Provider list -> ``[{"name","label"}]``; chat-capable models only, and only the FREE ones on OpenRouter."""
    out = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        model_id = item["id"]
        label = item.get("display_name") or item.get("name") or model_id
        if provider == "gemini":
            model_id = model_id.removeprefix("models/")
            if "gemini" not in model_id and "gemma" not in model_id:
                continue
        low = model_id.lower()
        if provider != "anthropic" and any(word in low for word in _SKIP_WORDS):
            continue
        if provider == "openrouter":
            pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
            if not (low.endswith(":free") or (pricing and all(str(v) in ("0", "0.0") for v in pricing.values()))):
                continue
        out.append({"name": make_name(provider, model_id), "label": str(label)[:80]})
    out.sort(key=lambda m: m["label"].lower())
    return out


def supports_vision(name):
    """Whether images may be attached for this online model (a name heuristic; the provider rejects the rest)."""
    try:
        _, model_id = split_name(name)
    except OnlineError:
        return False
    return any(hint in model_id.lower() for hint in _VISION_HINTS)


def limits(name):
    """``(working window tokens, answer tokens)`` for this model's provider: free tiers get small windows."""
    try:
        provider, _ = split_name(name)
    except OnlineError:
        return 16384, DEFAULT_MAX_TOKENS
    info = PROVIDERS[provider]
    return info["ctx"], info["predict"]


def list_models(provider="anthropic", refresh=False):
    """``{"models": [{"name","label"}], "source": "api"|"default", "error": str|None}`` (needs a saved key)."""
    key = get_key(provider)
    if not key:
        return {"models": [], "source": "none", "error": None}
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(provider)
        if hit and not refresh and now < hit[0]:
            return hit[1]
    path = "/v1/models?limit=100" if PROVIDERS[provider]["style"] == "anthropic" else "/models"
    request = Request(f"{PROVIDERS[provider]['base']}{path}", headers=_headers(provider, key), method="GET")
    try:
        with urlopen(request, timeout=15) as response:
            items = json.loads(response.read().decode("utf-8")).get("data", [])
        models = _parse_models(provider, items)
        if not models:
            raise ValueError("empty model list")
        result = {"models": models, "source": "api", "error": None}
        ttl = _LIST_TTL
    except HTTPError as exc:
        error = _explain_http_error(exc)
        # a rejected key must not look like a working one; other failures fall back to the default list
        fallback = [] if exc.code in (401, 403) else [
            {"name": make_name(provider, i), "label": label} for i, label in FALLBACK_MODELS[provider]]
        result = {"models": fallback, "source": "default" if fallback else "none", "error": error.message}
        ttl = 30.0
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        result = {"models": [{"name": make_name(provider, i), "label": label} for i, label in FALLBACK_MODELS[provider]],
                  "source": "default", "error": _network_error(exc).message}
        ttl = 30.0
    with _LOCK:
        _CACHE[provider] = (now + ttl, result)
    return result


def all_models():
    """Online models of every provider that has a key (used by the model selector)."""
    out = []
    for provider in PROVIDERS:
        out += list_models(provider)["models"]
    return out


# ---------------------------------------------------------------------------------------------- request building
def _media_type(b64):
    head = b64[:16]
    if head.startswith("/9j/"):
        return "image/jpeg"
    if head.startswith("iVBOR"):
        return "image/png"
    if head.startswith("R0lG"):
        return "image/gif"
    if head.startswith("UklG"):
        return "image/webp"
    return "image/png"


def build_request(ollama_payload):
    """Ollama-style chat payload -> (provider, request body) in the provider's own format."""
    provider, model_id = split_name(ollama_payload.get("model"))
    if PROVIDERS[provider]["style"] == "openai":
        return provider, _build_openai(provider, model_id, ollama_payload)
    options = ollama_payload.get("options") or {}
    system_parts, turns = [], []
    for message in ollama_payload.get("messages") or []:
        role, text = message.get("role"), str(message.get("content") or "")
        if role == "system":
            if text.strip():
                system_parts.append(text)
            continue
        if role == "tool":                      # tools are not offered to online models: keep the text as context
            role, text = "user", f"[نتیجهٔ ابزار]\n{text}"
        if role not in ("user", "assistant"):
            continue
        blocks = []
        for image in message.get("images") or []:
            if isinstance(image, str) and image:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": _media_type(image), "data": image}})
        if text.strip():
            blocks.append({"type": "text", "text": text})
        if not blocks:
            continue                            # the API rejects empty messages
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += blocks      # consecutive same-role turns must be merged
        else:
            turns.append({"role": role, "content": blocks})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)                            # the conversation must start with a user turn
    if not turns:
        raise OnlineError("پیامی برای ارسال به مدل آنلاین وجود ندارد.")
    body = {"model": model_id, "messages": turns, "stream": True,
            "max_tokens": int(options.get("num_predict") or DEFAULT_MAX_TOKENS)}
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if isinstance(options.get("temperature"), (int, float)):
        body["temperature"] = max(0.0, min(1.0, float(options["temperature"])))
    return provider, body


def _build_openai(provider, model_id, ollama_payload):
    """OpenAI-compatible ``/chat/completions`` body (Gemini, Groq, OpenRouter). Tools are never offered."""
    options = ollama_payload.get("options") or {}
    turns = []
    for message in ollama_payload.get("messages") or []:
        role, text = message.get("role"), str(message.get("content") or "")
        if role == "tool":
            role, text = "user", f"[نتیجهٔ ابزار]\n{text}"
        if role not in ("system", "user", "assistant"):
            continue
        images = [i for i in (message.get("images") or []) if isinstance(i, str) and i] if role == "user" else []
        if not text.strip() and not images:
            continue
        if turns and turns[-1]["role"] == role and role != "system":
            turns[-1]["texts"].append(text)         # consecutive same-role turns are merged
            turns[-1]["images"] += images
        else:
            turns.append({"role": role, "texts": [text] if text.strip() else [], "images": images})
    while turns and turns[0]["role"] == "assistant":
        turns.pop(0)
    if not any(t["role"] == "user" for t in turns):
        raise OnlineError("پیامی برای ارسال به مدل آنلاین وجود ندارد.")
    system = "\n\n".join("\n".join(t["texts"]) for t in turns if t["role"] == "system")
    messages = [{"role": "system", "content": system}] if system.strip() else []
    for turn in turns:
        if turn["role"] == "system":
            continue
        text = "\n\n".join(turn["texts"])
        if turn["images"]:
            parts = [{"type": "image_url", "image_url": {"url": f"data:{_media_type(i)};base64,{i}"}} for i in turn["images"]]
            if text:
                parts.append({"type": "text", "text": text})
            messages.append({"role": turn["role"], "content": parts})
        else:
            messages.append({"role": turn["role"], "content": text})
    body = {"model": model_id, "messages": messages, "stream": True,
            "max_tokens": int(options.get("num_predict") or PROVIDERS[provider]["predict"])}
    if PROVIDERS[provider].get("usage"):
        body["stream_options"] = {"include_usage": True}
    if isinstance(options.get("temperature"), (int, float)):
        body["temperature"] = max(0.0, min(1.5, float(options["temperature"])))
    return body


# ---------------------------------------------------------------------------------------------- streaming
class _Stream:
    """Iterates NDJSON lines (bytes) in Ollama's shape from an Anthropic SSE response; ``close`` aborts the request."""

    def __init__(self, response, model_name):
        self._response, self._model = response, model_name
        self.fp = getattr(response, "fp", None)

    def close(self):
        try:
            self._response.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _line(self, obj):
        return json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"

    def __iter__(self):
        usage_in = usage_out = 0
        reason = "stop"
        event = None
        done = False
        for raw in self._response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[5:].strip())
            except ValueError:
                continue
            kind = data.get("type") or event
            if kind == "message_start":
                usage_in = int(((data.get("message") or {}).get("usage") or {}).get("input_tokens") or 0)
            elif kind == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield self._line({"model": self._model, "message": {"role": "assistant", "content": delta["text"]},
                                      "done": False})
                elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                    yield self._line({"model": self._model, "message": {"role": "assistant", "content": "",
                                                                        "thinking": delta["thinking"]}, "done": False})
            elif kind == "message_delta":
                stop = (data.get("delta") or {}).get("stop_reason")
                reason = {"max_tokens": "length", "end_turn": "stop", "stop_sequence": "stop"}.get(stop, stop or "stop")
                usage_out = int((data.get("usage") or {}).get("output_tokens") or usage_out)
            elif kind == "error":
                error = data.get("error") or {}
                raise OnlineError(f"سرویس آنلاین خطا داد: {str(error.get('message') or error)[:300]}")
            elif kind == "message_stop":
                done = True
                break
        if not done:
            raise OnlineError("اتصال به سرویس آنلاین پیش از پایان پاسخ قطع شد.")
        yield self._line({"model": self._model, "message": {"role": "assistant", "content": ""}, "done": True,
                          "done_reason": reason, "prompt_eval_count": usage_in, "eval_count": usage_out})


class _OpenAIStream(_Stream):
    """Same NDJSON output, from an OpenAI-style ``data: {chunk}`` / ``data: [DONE]`` stream."""

    def __iter__(self):
        usage_in = usage_out = 0
        reason, done = "stop", False
        for raw in self._response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue                                # comments (": OPENROUTER PROCESSING") and blank lines
            body = line[5:].strip()
            if body == "[DONE]":
                done = True
                break
            try:
                data = json.loads(body)
            except ValueError:
                continue
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                continue
            if data.get("error"):
                error = data["error"]
                raise OnlineError(f"سرویس آنلاین خطا داد: {str(error.get('message') if isinstance(error, dict) else error)[:300]}")
            usage = data.get("usage")
            if isinstance(usage, dict):
                usage_in = int(usage.get("prompt_tokens") or usage_in)
                usage_out = int(usage.get("completion_tokens") or usage_out)
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield self._line({"model": self._model, "message": {"role": "assistant", "content": delta["content"]},
                                      "done": False})
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(thought, str) and thought:
                    yield self._line({"model": self._model, "message": {"role": "assistant", "content": "",
                                                                        "thinking": thought}, "done": False})
                if choice.get("finish_reason"):
                    reason = {"length": "length", "stop": "stop"}.get(choice["finish_reason"], "stop")
        if not done:
            raise OnlineError("اتصال به سرویس آنلاین پیش از پایان پاسخ قطع شد.")
        yield self._line({"model": self._model, "message": {"role": "assistant", "content": ""}, "done": True,
                          "done_reason": reason, "prompt_eval_count": usage_in, "eval_count": usage_out})


def open_stream(ollama_payload, timeout=600):
    """Open a streaming reply for an online model. Raises OnlineError with a Persian message on any failure."""
    provider, body = build_request(ollama_payload)
    key = get_key(provider)
    if not key:
        raise OnlineError("کلید API برای این ارائه‌دهنده ثبت نشده است؛ در «مدیریت مدل‌ها ← آنلاین» ثبتش کن.")
    openai = PROVIDERS[provider]["style"] == "openai"
    request = Request(f"{PROVIDERS[provider]['base']}{'/chat/completions' if openai else '/v1/messages'}",
                      data=json.dumps(body).encode("utf-8"), headers=_headers(provider, key), method="POST")
    try:
        return (_OpenAIStream if openai else _Stream)(urlopen(request, timeout=timeout), ollama_payload.get("model"))
    except HTTPError as exc:
        raise _explain_http_error(exc) from None
    except (URLError, TimeoutError, OSError) as exc:
        raise _network_error(exc) from None


def complete(ollama_payload, timeout=120):
    """Non-streaming helper for internal calls (titles, summaries): Ollama-shaped ``{"message": {"content": ...}}``."""
    text = []
    with open_stream(ollama_payload, timeout) as stream:
        for line in stream:
            packet = json.loads(line.decode("utf-8"))
            text.append((packet.get("message") or {}).get("content") or "")
    return {"message": {"role": "assistant", "content": "".join(text)}, "done": True}
