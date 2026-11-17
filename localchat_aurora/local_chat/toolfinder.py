"""Find developer tools that are installed but not on PATH (Node/npm, Git).

Only *finding* and *version probing* happen here; nothing is installed or modified.
A user can pin a tool by writing ``tool_paths.json`` into the LocalChat data folder::

    {"npm": "D:\\\\nodejs\\\\npm.cmd"}
"""
import glob
import json
import os
import re
import shutil
import threading
from pathlib import Path

if __package__:
    from . import settings as app_settings
    from .procutil import run_capture
else:
    import settings as app_settings
    from procutil import run_capture

WINDOWS = os.name == "nt"
EXTENSIONS = (".exe", ".cmd", ".bat", ".com") if WINDOWS else ("",)

# command names on PATH, per logical tool
NAMES = {
    "node": ("node",),
    "npm": ("npm",),
    "git": ("git",),
    "py": ("py",),
}
VERSION_ARGS = {
    "node": ["--version"], "npm": ["--version"], "git": ["--version"], "py": ["--list"],
}
_CACHE = {"time": 0.0, "data": {}}
_LOCK = threading.RLock()


def _env(name):
    return os.environ.get(name) or ""


def _home():
    return Path(os.path.expanduser("~"))


def _patterns(tool):
    """Candidate file patterns (may contain *) in addition to PATH."""
    home = _home()
    pf = [p for p in (_env("ProgramFiles"), _env("ProgramFiles(x86)"), _env("ProgramW6432")) if p]
    local = _env("LOCALAPPDATA")
    drives = [f"{letter}:\\" for letter in "CDEFG"] if WINDOWS else ["/opt", "/usr/local", str(home)]
    out = []

    def add(*parts):
        out.append(os.path.join(*parts))

    if tool in {"node", "npm"}:
        exe = ("node.exe" if tool == "node" else "npm.cmd") if WINDOWS else tool
        for base in pf:
            add(base, "nodejs", exe)
        if local:
            add(local, "Programs", "nodejs", exe)
        add(str(home), ".nvm", "current", exe)
        add(str(home), "scoop", "apps", "nodejs", "current", exe)
    elif tool == "git":
        for base in pf:
            add(base, "Git", "cmd", "git.exe")
        if local:
            add(local, "Programs", "Git", "cmd", "git.exe")
    return out


def _configured():
    path = app_settings.app_data_dir() / "tool_paths.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def find(tool):
    """Absolute path of ``tool`` (see NAMES) or ``None``. Order: user setting, PATH, well-known folders."""
    if tool not in NAMES:
        return None
    pinned = _configured().get(tool)
    if isinstance(pinned, str) and pinned.strip():
        candidate = Path(pinned.strip().strip('"'))
        if candidate.is_file():
            return str(candidate)
    for name in NAMES[tool]:
        found = shutil.which(name)
        if found:
            return found
    for pattern in _patterns(tool):
        if "*" in pattern:
            matches = sorted(glob.glob(pattern), reverse=True)  # newest version folder first
            for match in matches:
                if os.path.isfile(match):
                    return match
        elif os.path.isfile(pattern):
            return pattern
    return None


def env_with_dir(exe, env=None):
    """Environment whose PATH also contains the tool's folder (some tools need helpers that sit next to them)."""
    merged = dict(os.environ if env is None else env)
    folder = str(Path(exe).parent)
    merged["PATH"] = folder + os.pathsep + merged.get("PATH", "")
    return merged


def _first_line(text):
    lines = [line.strip() for line in (text or "").splitlines()
             if line.strip() and not line.strip().startswith("Picked up") and re.search(r"[A-Za-z0-9]", line)]
    for line in lines:
        if re.search(r"\d+\.\d+", line):
            return line[:160]
    return lines[0][:160] if lines else ""


def _probe(tool, exe):
    info = {"found": True, "path": exe, "on_path": bool(shutil.which(NAMES[tool][0])) and
            os.path.normcase(shutil.which(NAMES[tool][0]) or "") == os.path.normcase(exe)}
    args = VERSION_ARGS.get(tool)
    if not args:
        return info
    try:
        result = run_capture([exe, *args], timeout=20, env=env_with_dir(exe))
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        info["version"] = _first_line(result.stdout)
        if result.returncode != 0 and not info["version"]:
            info["version"] = "(اجرا شد ولی نسخه خوانده نشد)"
    except OSError as exc:
        info["version"] = f"(اجرا نشد: {str(exc)[:80]})"
    return info


def scan(force=False, tools=None, max_age=120):
    """Dict tool → {found, path, version, on_path}; probes run in parallel and are cached for ``max_age`` seconds."""
    import time as _time
    now = _time.monotonic()
    with _LOCK:
        if not force and _CACHE["data"] and now - _CACHE["time"] < max_age and tools is None:
            return dict(_CACHE["data"])
    names = list(tools or NAMES)
    result = {}

    def work(tool):
        exe = find(tool)
        result[tool] = _probe(tool, exe) if exe else {"found": False}

    threads = [threading.Thread(target=work, args=(tool,), daemon=True) for tool in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=45)
    for tool in names:
        result.setdefault(tool, {"found": False, "version": "(بررسی زمان‌بر شد)"})
    if tools is None:
        with _LOCK:
            _CACHE.update(time=now, data=dict(result))
    return result


def describe(found):
    """Short Persian summary for prompts / the environment report."""
    parts = []
    for tool, info in found.items():
        if info.get("found"):
            parts.append(f"{tool}: {info.get('version') or 'نصب است'}" + ("" if info.get("on_path") else " (خارج از PATH)"))
    missing = [tool for tool, info in found.items() if not info.get("found")]
    text = "؛ ".join(parts) or "هیچ ابزاری پیدا نشد"
    if missing:
        text += " | پیدا نشد: " + ", ".join(missing)
    return text
