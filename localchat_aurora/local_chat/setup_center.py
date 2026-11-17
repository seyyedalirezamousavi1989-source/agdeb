"""Setup Center: everything the app needs besides itself, detected and installed from inside the app.

* ``status()``          – what is installed / missing (Ollama, WebView2, audio engine, Whisper, Edge TTS, ParsVoice).
* ``recommend()``       – which local models fit THIS computer (VRAM / RAM), best first.
* ``start_job()``       – background installs with automatic resume: Ollama, the audio engine, speech/LLM models.
* ``set_path()``        – the user points the app at things already installed (Ollama, audio Python, ParsVoice folder, models folder).
* ``missing_for()``     – which component a request needs (chat, voice, ...) so the UI can say "X is not installed" only then.

Downloads are HTTP-Range resumable and retried with back-off; a job that was running when the app closed is resumed at the next start.
Network access happens only inside jobs the user started; nothing is fetched at import time.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from . import settings as app_settings
    from . import hardware
    from . import models_catalog
except ImportError:  # pragma: no cover - flat import layout
    import settings as app_settings
    import hardware
    import models_catalog

OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_SETUP_URL = "https://ollama.com/download/OllamaSetup.exe"
PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
VC_REDIST_URL = "https://aka.ms/vc14/vc_redist.x64.exe"
WEBVIEW2_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
AUDIO_PACKAGES = ("faster-whisper", "edge-tts", "huggingface_hub")
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")
GB = 1024 ** 3
MAX_ATTEMPTS = 40            # per step; with the capped back-off this rides out hours of a bad connection
BACKOFF_CAP = 60
CHUNK = 1024 * 256
PATH_KEYS = {
    "ollama_exe": ("file", "مسیر ollama.exe"),
    "models_dir": ("dir", "پوشهٔ مدل‌های Ollama"),
    "audio_python": ("file", "مسیر python.exe محیط صدا (faster-whisper / TTS)"),
    "parsvoice_dir": ("dir", "پوشهٔ مدل ParsVoice (config.json و model.pth)"),
}


# ----------------------------------------------------------------------------------------------- locations
def data_dir():
    folder = app_settings.app_data_dir() / "setup"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def voice_root():
    return app_settings.app_data_dir() / "voice"


def audio_python_dir():
    return voice_root() / "audio-python"


def _jobs_file():
    return data_dir() / "jobs.json"


# ----------------------------------------------------------------------------------------------- user paths
def get_paths():
    value = app_settings.load_settings().get("setup_paths")
    return {k: v for k, v in (value.items() if isinstance(value, dict) else []) if k in PATH_KEYS and isinstance(v, str)}


def set_path(key, raw):
    """Store (or clear, with '') a path of something already installed. Raises ValueError with a Persian message."""
    if key not in PATH_KEYS:
        raise ValueError("این مسیر شناخته‌شده نیست.")
    kind, label = PATH_KEYS[key]
    value = raw.strip().strip('"') if isinstance(raw, str) else None
    if value is None or len(value) > 500 or "\x00" in value:
        raise ValueError("مسیر معتبر نیست.")
    if key == "models_dir":
        app_settings.set_models_dir(value)   # existing validation + Ollama restart hint live there
        return get_paths()
    data = app_settings.load_settings()
    paths = data.get("setup_paths") if isinstance(data.get("setup_paths"), dict) else {}
    if not value:
        paths.pop(key, None)
    else:
        target = Path(os.path.expandvars(value)).expanduser()
        if not target.is_absolute():
            raise ValueError("مسیر باید کامل باشد، مثلاً D:\\Tools\\python.exe")
        if (kind == "file" and not target.is_file()) or (kind == "dir" and not target.is_dir()):
            raise ValueError(f"{label} پیدا نشد.")
        if key == "parsvoice_dir" and not ((target / "config.json").is_file() and (target / "model.pth").is_file()):
            raise ValueError("در این پوشه config.json و model.pth پیدا نشد.")
        paths[key] = str(target)
    data["setup_paths"] = paths
    app_settings._save_settings(data)
    apply_env()
    return get_paths()


def apply_env():
    """Export the user's paths the way the voice code already expects them (safe to call repeatedly)."""
    paths = get_paths()
    if paths.get("audio_python"):
        os.environ["AIRESEARCH_STT_PYTHON"] = paths["audio_python"]
        os.environ["AIRESEARCH_TTS_PYTHON"] = paths["audio_python"]
    if paths.get("parsvoice_dir"):
        os.environ["AIRESEARCH_PARVOICE_DIR"] = paths["parsvoice_dir"]
    stt = app_settings.load_settings().get("stt_model")
    if isinstance(stt, str) and stt in WHISPER_SIZES:
        os.environ.setdefault("LOCALCHAT_STT_MODEL", stt)


def stt_model():
    value = app_settings.load_settings().get("stt_model")
    return value if value in WHISPER_SIZES else ""


# ----------------------------------------------------------------------------------------------- detection
def find_ollama():
    configured = get_paths().get("ollama_exe")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                      Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
                      Path("C:/Program Files/Ollama/ollama.exe")):
        if candidate.is_file():
            return str(candidate)
    return ""


def ollama_running(timeout=2):
    try:
        with urlopen(f"{OLLAMA_API}/api/tags", timeout=timeout):
            return True
    except (OSError, URLError, ValueError):
        return False


def start_ollama(wait=40):
    """Start `ollama serve` (with the chosen models folder) and wait until it answers."""
    if ollama_running():
        return True
    exe = find_ollama()
    if not exe:
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([exe, "serve"], env=app_settings.ollama_environment(), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if ollama_running():
            return True
        time.sleep(1)
    return False


def _webview2_version():
    if os.name != "nt":
        return None
    try:
        import winreg
        key = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(root, key) as handle:
                    return winreg.QueryValueEx(handle, "pv")[0]
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return ""


def _vc_redist_installed():
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64") as handle:
            return winreg.QueryValueEx(handle, "Installed")[0] == 1
    except OSError:
        return False
    except Exception:  # noqa: BLE001
        return False


def audio_python_candidates():
    paths = get_paths()
    items = [paths.get("audio_python", ""), os.environ.get("AIRESEARCH_STT_PYTHON", ""),
             str(audio_python_dir() / "python.exe"), str(voice_root() / ".venv-parsvoice" / "Scripts" / "python.exe")]
    seen, result = set(), []
    for item in items:
        if item and os.path.normcase(item) not in seen and Path(item).is_file():
            seen.add(os.path.normcase(item))
            result.append(item)
    return result


def has_package(python, package, timeout=20):
    try:
        return subprocess.run([python, "-c", f"import {package}"], capture_output=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _audio_package_state(package):
    for python in audio_python_candidates():
        if has_package(python, package):
            return python
    return ""


def parsvoice_dir():
    candidates = [get_paths().get("parsvoice_dir", ""), os.environ.get("AIRESEARCH_PARVOICE_DIR", ""),
                  str(voice_root() / "models" / "ParsVoice-XTTS")]
    for item in candidates:
        if item and (Path(item) / "config.json").is_file() and (Path(item) / "model.pth").is_file():
            return item
    return ""


def status(deep=True):
    """Every component with ok/missing, what it is for and how to get it. ``deep=False`` skips the slow import probes."""
    exe, running = find_ollama(), ollama_running()
    whisper = _audio_package_state("faster_whisper") if deep else ""
    edge = _audio_package_state("edge_tts") if deep else ""
    parsvoice = parsvoice_dir()
    webview = _webview2_version()
    vc = _vc_redist_installed()
    items = [
        {"id": "ollama", "label": "Ollama (موتور مدل‌های محلی)", "ok": bool(exe) and running, "installed": bool(exe),
         "detail": ("در حال اجرا" if running else "نصب است ولی اجرا نیست" if exe else "نصب نیست") + (f" — {exe}" if exe else ""),
         "needed_for": "گفت‌وگو با مدل‌های محلی", "install": "ollama" if not exe else None, "start": bool(exe) and not running,
         "path_key": "ollama_exe"},
        {"id": "audio_env", "label": "موتور صدا (Python + faster-whisper + Edge TTS)", "ok": bool(whisper),
         "detail": f"آماده — {whisper}" if whisper else "نصب نیست", "needed_for": "ورودی صوتی/میکروفون و خواندن پاسخ‌ها",
         "install": "audio_env" if not whisper else None, "path_key": "audio_python"},
        {"id": "edge_tts", "label": "صدای آنلاین Edge TTS", "ok": bool(edge), "detail": "آماده" if edge else "نصب نیست",
         "needed_for": "خواندن پاسخ با صدای فارسی (نیازمند اینترنت)", "install": "audio_env" if not edge else None},
        {"id": "parsvoice", "label": "مدل صدای آفلاین ParsVoice (اختیاری)", "ok": bool(parsvoice), "optional": True,
         "detail": parsvoice or "پیدا نشد — اختیاری؛ اگر دارید مسیرش را بدهید", "needed_for": "خواندن پاسخ بدون اینترنت",
         "install": None, "path_key": "parsvoice_dir",
         "hint": "این مدل نیاز به پذیرش مجوز در Hugging Face و محیط جداگانهٔ Coqui TTS دارد؛ نصب خودکار ندارد."},
        {"id": "vc_redist", "label": "Microsoft Visual C++ Runtime", "ok": vc is None or bool(vc), "optional": True,
         "detail": "لازم نیست" if vc is None else "نصب است" if vc else "نصب نیست؛ اگر برنامه یا کتابخانه‌ای خطای DLL داد نصبش کنید",
         "needed_for": "اجرای درست کتابخانه‌های برنامه (نمودار، MT5، صدا)", "install": "vc_redist" if vc is False else None,
         "why": "بستهٔ رسمی مایکروسافت که بیشتر برنامه‌ها به آن نیاز دارند؛ حدود ۲۵ مگابایت."},
        {"id": "webview2", "label": "Microsoft Edge WebView2 (پنجرهٔ برنامه)", "ok": webview is None or bool(webview), "optional": True,
         "detail": "لازم نیست" if webview is None else (webview or "نصب نیست؛ برنامه در مرورگر باز می‌شود"),
         "needed_for": "نمایش برنامه در پنجرهٔ مستقل به‌جای مرورگر", "install": "webview2" if webview == "" else None,
         "why": "اگر نصب نباشد برنامه در مرورگر شما باز می‌شود و همه‌چیز کار می‌کند؛ نصبش فقط ظاهر برنامه را مستقل می‌کند."},
    ]
    return {"components": items, "paths": get_paths(), "models_dir": str(app_settings.effective_models_dir()),
            "stt_model": stt_model() or os.environ.get("LOCALCHAT_STT_MODEL", "small")}


def missing_for(feature):
    """For a request that needs `feature` ('chat', 'voice_in', 'voice_out', 'model'), the missing component or None."""
    if feature in {"chat", "model"}:
        if not find_ollama() and not ollama_running():
            return {"component": "ollama", "message": "Ollama نصب نیست. از «مرکز راه‌اندازی» آن را نصب کنید یا مسیرش را بدهید."}
        if not ollama_running():
            return {"component": "ollama", "message": "Ollama نصب است ولی اجرا نیست؛ از «مرکز راه‌اندازی» روشنش کنید."}
    if feature == "voice_in" and not _audio_package_state("faster_whisper"):
        return {"component": "audio_env", "message": "موتور تشخیص گفتار نصب نیست. از «مرکز راه‌اندازی» نصبش کنید."}
    if feature == "voice_out" and not _audio_package_state("edge_tts"):
        return {"component": "audio_env", "message": "موتور صدا نصب نیست. از «مرکز راه‌اندازی» نصبش کنید."}
    return None


# ----------------------------------------------------------------------------------------------- recommendations
def size_gb(text):
    """Largest number in strings like '≈ 6.6–7.6 GB' / '≈ 700 MB' → gigabytes."""
    numbers = re.findall(r"\d+(?:\.\d+)?", str(text or ""))
    if not numbers:
        return 0.0
    value = max(float(n) for n in numbers)
    return value / 1024 if re.search(r"\bMB\b", str(text), re.I) else value


def classify_machine(snapshot=None):
    """{'mode': 'gpu'|'cpu', 'vram_gb', 'ram_gb', 'gpu_name', 'fit_gb', 'offload_gb'}.

    fit_gb: a model of this size (weights) runs fully on the GPU with room for the context; offload_gb: still usable by
    spilling layers into RAM (slower). On CPU-only machines both come from RAM."""
    snap = snapshot if snapshot is not None else hardware.snapshot()
    ram = (snap.get("ram") or {}).get("total")
    ram_gb = ram / GB if isinstance(ram, (int, float)) else 0.0
    gpus = [g for g in (snap.get("gpus") or []) if g.get("dedicated") and isinstance(g.get("total"), int)]
    best = max(gpus, key=lambda g: g["total"]) if gpus else None
    if best and best["total"] >= 3 * GB:
        vram = best["total"] / GB
        return {"mode": "gpu", "vram_gb": round(vram, 1), "ram_gb": round(ram_gb, 1), "gpu_name": best["name"],
                "fit_gb": round(vram * 0.78, 1), "offload_gb": round(vram * 0.78 + ram_gb * 0.35, 1)}
    return {"mode": "cpu", "vram_gb": 0.0, "ram_gb": round(ram_gb, 1), "gpu_name": (best or {}).get("name", ""),
            "fit_gb": round(ram_gb * 0.35, 1), "offload_gb": round(ram_gb * 0.5, 1)}


def whisper_for(machine):
    if machine["mode"] == "gpu":
        return "medium" if machine["vram_gb"] >= 8 else "small"
    return "small" if machine["ram_gb"] >= 12 else "base"


def recommend(snapshot=None, installed=()):
    """Catalog ranked for this machine: every entry gets fit = 'good' | 'slow' | 'too_big'; groups list best first."""
    machine = classify_machine(snapshot)
    have = {models_catalog.normalize_name(n) for n in installed or []}
    entries = []
    for item in models_catalog.CATALOG:
        gb = size_gb(item.get("size"))
        fit = "good" if gb <= machine["fit_gb"] else "slow" if gb <= machine["offload_gb"] else "too_big"
        entries.append({**item, "size_gb": round(gb, 1), "fit": fit, "installed": models_catalog.normalize_name(item["id"]) in have})
    rank = {"good": 0, "slow": 1, "too_big": 2}
    groups = {}
    for group in ("text", "vision", "audio", "embedding"):
        members = [e for e in entries if e["group"] == group]
        # fits first; among those that fit, the biggest = the strongest in this catalog; ties prefer the curated "recommended"
        members.sort(key=lambda e: (rank[e["fit"]], -e["size_gb"] if e["fit"] != "too_big" else e["size_gb"], not e.get("recommended")))
        groups[group] = members
    starter = []
    for group in ("text", "vision", "embedding"):
        pick = next((e for e in groups[group] if e["fit"] != "too_big"), None)
        if pick and not (group == "vision" and any(e.get("vision") for e in starter)):
            starter.append(pick)
    return {"machine": machine, "stt_model": whisper_for(machine), "groups": groups, "starter": starter,
            "note": ("کارت گرافیک اختصاصی پیدا نشد؛ مدل‌ها روی CPU اجرا می‌شوند و کندترند. مدل‌های کوچک‌تر را انتخاب کنید."
                     if machine["mode"] == "cpu" else
                     f"مدل‌های «good» کاملاً روی GPU ({machine['vram_gb']} GB) جا می‌شوند؛ «slow» بخشی از لایه‌ها را روی RAM می‌برند.")}


# ----------------------------------------------------------------------------------------------- resumable download
class Cancelled(Exception):
    pass


class DownloadError(Exception):
    pass


def download_file(url, dest, progress=None, cancel=None, attempts=MAX_ATTEMPTS, sleep=time.sleep, timeout=30):
    """Download `url` to `dest` resuming a leftover `.part` with HTTP Range; retries with back-off. Returns the final Path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    delay = 2
    for attempt in range(1, attempts + 1):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        have = part.stat().st_size if part.is_file() else 0
        request = Request(url, headers={"User-Agent": "LocalChat-Setup", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urlopen(request, timeout=timeout) as response:
                status_code = getattr(response, "status", 200)
                length = int(response.headers.get("Content-Length") or 0)
                if have and status_code != 206:          # server ignored Range: start over
                    have = 0
                total = have + length if length else 0
                with open(part, "ab" if have else "wb") as handle:
                    done = have
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
            if total and part.stat().st_size < total:
                raise OSError("اتصال زودتر از موعد قطع شد")
            os.replace(part, dest)
            return dest
        except HTTPError as exc:
            if exc.code == 416 and part.is_file():       # already complete
                os.replace(part, dest)
                return dest
            if exc.code in (400, 401, 403, 404, 410):
                raise DownloadError(f"سرور دانلود را رد کرد (HTTP {exc.code}).") from exc
            error = exc
        except (URLError, TimeoutError, OSError, ConnectionError) as exc:
            error = exc
        if progress:
            progress(None, None, f"اتصال قطع شد؛ تلاش دوباره ({attempt}/{attempts}) …")
        sleep(min(delay, BACKOFF_CAP))
        delay = min(delay * 2, BACKOFF_CAP)
    raise DownloadError(f"دانلود پس از چند تلاش کامل نشد: {error}")


# ----------------------------------------------------------------------------------------------- jobs
_JOBS = {}
_LOCK = threading.RLock()
_RUN = threading.Lock()          # one install at a time: they all share the same connection and disk
_CANCEL = {}


def _persist():
    with _LOCK:
        keep = [j for j in _JOBS.values()]
        try:
            _jobs_file().write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass


def _update(job, **changes):
    with _LOCK:
        job.update(changes)
        job["updated"] = time.time()
    _persist()


def jobs():
    with _LOCK:
        return [dict(j) for j in sorted(_JOBS.values(), key=lambda j: j["created"], reverse=True)]


def cancel_job(job_id):
    event = _CANCEL.get(job_id)
    if event:
        event.set()
    with _LOCK:
        job = _JOBS.get(job_id)
        if job and job["state"] in {"queued", "running", "waiting"}:
            job["state"] = "cancelled"
            job["message"] = "لغو شد؛ دفعهٔ بعد از همین‌جا ادامه می‌دهد."
    _persist()
    return bool(event)


def _run_cmd(command, cancel, cwd=None, line=None):
    """Run a subprocess, streaming lines to `line`; killed when cancelled."""
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for text in process.stdout:
            if cancel.is_set():
                process.kill()
                raise Cancelled()
            if line and text.strip():
                line(text.strip()[-200:])
        return process.wait()
    finally:
        if process.poll() is None:
            process.kill()


def _step(job, cancel, label, func, attempts=3):
    """Run `func` retrying on failure (each retry resumes: downloads use Range, pip reuses its cache)."""
    last = None
    for attempt in range(1, attempts + 1):
        if cancel.is_set():
            raise Cancelled()
        _update(job, state="running", message=label if attempt == 1 else f"{label} (تلاش {attempt}/{attempts})")
        try:
            return func()
        except Cancelled:
            raise
        except (DownloadError, RuntimeError, OSError) as exc:
            last = exc
            _update(job, state="waiting", message=f"{label}: {exc}؛ تا چند ثانیه دیگر دوباره تلاش می‌شود")
            for _ in range(min(5 * attempt, 20)):
                if cancel.is_set():
                    raise Cancelled()
                time.sleep(1)
    raise DownloadError(str(last))


def _file_progress(job):
    last = {"t": 0.0}

    def report(done, total, message=None):
        if message:
            _update(job, state="waiting", message=message)
            return
        now = time.monotonic()
        if now - last["t"] < 0.4:
            return
        last["t"] = now
        pct = round(done * 100 / total, 1) if total else None
        _update(job, state="running", bytes=done, total=total or job.get("total", 0), progress=pct if pct is not None else job.get("progress", 0))
    return report


def _install_ollama(job, cancel):
    if find_ollama():
        start_ollama()
        return
    installer = _step(job, cancel, "دانلود نصب‌کنندهٔ Ollama",
                      lambda: download_file(OLLAMA_SETUP_URL, data_dir() / "downloads" / "OllamaSetup.exe", _file_progress(job), cancel))
    _update(job, message="نصب Ollama …", progress=100)

    def run_installer():
        code = _run_cmd([str(installer), "/SILENT", "/NORESTART", "/SUPPRESSMSGBOXES"], cancel)
        if code != 0 and not find_ollama():
            raise RuntimeError(f"نصب Ollama با کد {code} پایان یافت")
    _step(job, cancel, "نصب Ollama", run_installer, attempts=2)
    deadline = time.monotonic() + 60
    while not find_ollama() and time.monotonic() < deadline:
        time.sleep(1)
    if not find_ollama():
        raise DownloadError("Ollama نصب شد ولی پیدا نشد؛ مسیر ollama.exe را دستی بدهید.")
    _update(job, message="روشن‌کردن Ollama …")
    start_ollama()


def _install_ms_runtime(job, cancel, url, filename, args, label):
    """Official Microsoft redistributable: resumable download, silent install (shows the Windows admin prompt once)."""
    if os.name != "nt":
        raise DownloadError("این بسته فقط برای ویندوز است.")
    installer = _step(job, cancel, f"دانلود {label}", lambda: download_file(url, data_dir() / "downloads" / filename, _file_progress(job), cancel))
    _update(job, message=f"نصب {label} …", progress=100)

    def run():
        code = _run_cmd([str(installer), *args], cancel)
        if code not in (0, 1638, 3010):          # 1638: newer version already present, 3010: restart suggested
            raise RuntimeError(f"نصب {label} با کد {code} پایان یافت")
    _step(job, cancel, f"نصب {label}", run, attempts=2)


def _install_audio_env(job, cancel):
    target = audio_python_dir()
    python = target / "python.exe"
    if os.name != "nt":
        raise DownloadError("نصب خودکار موتور صدا فقط روی ویندوز پشتیبانی می‌شود؛ مسیر یک Python دارای faster-whisper را بدهید.")
    downloads = data_dir() / "downloads"
    if not python.is_file():
        archive = _step(job, cancel, "دانلود Python همراه (برای موتور صدا)",
                        lambda: download_file(PYTHON_EMBED_URL, downloads / "python-embed.zip", _file_progress(job), cancel))
        _update(job, message="استخراج Python …", progress=100)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
        for pth in target.glob("python*._pth"):             # enable site-packages so pip-installed libraries import
            text = pth.read_text(encoding="utf-8")
            pth.write_text(re.sub(r"^#\s*import site", "import site", text, flags=re.M), encoding="utf-8")
    if not has_package(str(python), "pip", 30):
        pip_script = _step(job, cancel, "دانلود نصب‌کنندهٔ pip",
                           lambda: download_file(GET_PIP_URL, downloads / "get-pip.py", _file_progress(job), cancel))

        def bootstrap():
            if _run_cmd([str(python), str(pip_script), "--no-warn-script-location"], cancel, line=lambda t: _update(job, message=t)) != 0:
                raise RuntimeError("نصب pip ناموفق بود")
        _step(job, cancel, "نصب pip", bootstrap)

    def install_packages():
        command = [str(python), "-m", "pip", "install", "--upgrade", "--retries", "10", "--timeout", "60",
                   "--no-warn-script-location", *AUDIO_PACKAGES]
        if _run_cmd(command, cancel, line=lambda t: _update(job, message=t)) != 0:
            raise RuntimeError("نصب کتابخانه‌های صدا ناموفق بود")
    _step(job, cancel, "نصب faster-whisper و Edge TTS (حجم زیاد؛ قطع شود ادامه می‌دهد)", install_packages, attempts=6)
    if not has_package(str(python), "faster_whisper", 60):
        raise DownloadError("faster-whisper نصب شد ولی import نشد.")
    apply_env()


def _install_whisper_model(job, cancel, size):
    python = (audio_python_candidates() or [""])[0]
    if not python:
        raise DownloadError("ابتدا «موتور صدا» را نصب کنید.")
    code = "import sys;from faster_whisper import download_model;print(download_model(sys.argv[1]))"

    def fetch():
        if _run_cmd([python, "-c", code, size], cancel, line=lambda t: _update(job, message=t)) != 0:
            raise RuntimeError("دریافت مدل گفتار ناموفق بود")
    _step(job, cancel, f"دانلود مدل گفتار «{size}» (در صورت قطع ادامه می‌دهد)", fetch, attempts=8)
    data = app_settings.load_settings()
    data["stt_model"] = size
    app_settings._save_settings(data)
    os.environ["LOCALCHAT_STT_MODEL"] = size


def _install_model(job, cancel, name):
    if not ollama_running() and not start_ollama():
        raise DownloadError("Ollama نصب یا اجرا نیست؛ اول آن را از «مرکز راه‌اندازی» نصب کنید.")

    def pull():
        request = Request(f"{OLLAMA_API}/api/pull", data=json.dumps({"model": name, "stream": True}).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=300) as response:
                for raw in response:
                    if cancel.is_set():
                        raise Cancelled()
                    try:
                        event = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        continue
                    if event.get("error"):
                        raise RuntimeError(str(event["error"])[:200])
                    total, done = event.get("total"), event.get("completed")
                    if total and done is not None:
                        _update(job, state="running", bytes=done, total=total, progress=round(done * 100 / total, 1),
                                message=event.get("status", "دانلود"))
                    elif event.get("status"):
                        _update(job, message=str(event["status"])[:120])
                    if event.get("status") == "success":
                        return
        except HTTPError as exc:
            raise DownloadError(f"Ollama مدل را رد کرد (HTTP {exc.code}).") from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError(f"اتصال قطع شد: {exc}") from exc
    # Ollama keeps finished layers, so every retry continues from where the last one stopped.
    _step(job, cancel, f"دانلود مدل {name}", pull, attempts=MAX_ATTEMPTS)
    _update(job, progress=100)


def _worker(job):
    cancel = _CANCEL.setdefault(job["id"], threading.Event())
    with _RUN:
        if cancel.is_set():
            return
        try:
            kind = job["kind"]
            if kind == "ollama":
                _install_ollama(job, cancel)
            elif kind == "audio_env":
                _install_audio_env(job, cancel)
            elif kind == "vc_redist":
                _install_ms_runtime(job, cancel, VC_REDIST_URL, "vc_redist.x64.exe", ["/install", "/quiet", "/norestart"], "Visual C++ Runtime")
            elif kind == "webview2":
                _install_ms_runtime(job, cancel, WEBVIEW2_URL, "MicrosoftEdgeWebView2Setup.exe", ["/silent", "/install"], "WebView2")
            elif kind.startswith("whisper:"):
                _install_whisper_model(job, cancel, kind.split(":", 1)[1])
            elif kind.startswith("model:"):
                _install_model(job, cancel, kind.split(":", 1)[1])
            else:
                raise DownloadError("نوع کار ناشناخته است.")
            _update(job, state="done", progress=100, message="انجام شد.")
        except Cancelled:
            _update(job, state="cancelled", message="لغو شد؛ دفعهٔ بعد از همین‌جا ادامه می‌دهد.")
        except Exception as exc:  # noqa: BLE001 - a job must end in a visible state
            _update(job, state="error", message=str(exc)[:300])


_KIND_OK = re.compile(r"^(ollama|audio_env|vc_redist|webview2|whisper:(?:%s)|model:[A-Za-z0-9._:/-]{1,100})$" % "|".join(WHISPER_SIZES))


def start_job(kind):
    """Start (or return the already running) job of this kind. Raises ValueError for an unknown kind."""
    if not isinstance(kind, str) or not _KIND_OK.fullmatch(kind):
        raise ValueError("نوع کار معتبر نیست.")
    with _LOCK:
        for job in _JOBS.values():
            if job["kind"] == kind and job["state"] in {"queued", "running", "waiting"}:
                return dict(job)
        job = {"id": uuid.uuid4().hex[:12], "kind": kind, "state": "queued", "progress": 0, "bytes": 0, "total": 0,
               "message": "در صف", "created": time.time(), "updated": time.time()}
        _JOBS[job["id"]] = job
    _persist()
    threading.Thread(target=_worker, args=(job,), daemon=True, name=f"setup-{kind}").start()
    return dict(job)


def resume_pending():
    """At app start: jobs that were running/waiting/queued when the app closed continue automatically."""
    try:
        saved = json.loads(_jobs_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    resumed = []
    for job in saved if isinstance(saved, list) else []:
        if not isinstance(job, dict) or not _KIND_OK.fullmatch(str(job.get("kind", ""))):
            continue
        if job.get("state") in {"queued", "running", "waiting"}:
            resumed.append(start_job(job["kind"]))
        else:
            with _LOCK:
                _JOBS.setdefault(job.get("id", uuid.uuid4().hex[:12]), job)
    return resumed
