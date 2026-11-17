"""Opt-in access from a phone / another device (PWA on iPhone) — and the rules that keep it safe.

LocalChat was built as a loopback-only app: anything that can reach it can talk to the agent, read
conversations and memory, start installs and shut the PC-side app down.  Opening the port to the network
therefore needs real authentication, not just a different bind address.  This module holds all of that
logic so server.py only has to ask «may this request pass?»:

* Remote access is OFF by default (bind 127.0.0.1).  It turns on with LOCALCHAT_REMOTE=1 (start_phone.bat)
  or ``"remote": {"enabled": true}`` in settings.json.
* A request is «local» only if it really comes from this PC: loopback peer, loopback Host header and no
  proxy headers.  Anything arriving through a reverse proxy / Tailscale Serve / a tunnel is «remote» even
  though its TCP peer is 127.0.0.1 — otherwise a proxy would silently bypass the login.
* Remote clients pair once with a long one-time-style pairing code (shown only on the PC at /phone), then
  keep a random device cookie.  Wrong codes are rate-limited globally.
* Remote clients can never use the endpoints that act on the PC itself (shut down, open folders / URLs,
  install software, change model paths, capture the screen …) and cannot manage pairing.
"""

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

from . import settings as app_settings

DEVICE_COOKIE = "lc_device"
DEVICE_COOKIE_MAX_AGE = 180 * 24 * 3600
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I: easy to type on a phone
CODE_GROUPS = 3
CODE_GROUP_LEN = 4
MAX_DEVICES = 20
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
# headers a reverse proxy / tunnel adds; their presence means the TCP peer is NOT the real client
PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-real-ip", "forwarded",
                 "via", "cf-connecting-ip", "tailscale-user-login", "tailscale-user-name")
HOST_SUFFIXES = (".local", ".lan", ".home.arpa", ".ts.net", ".internal")

# reachable without a device cookie (needed to log in / to install the PWA)
PUBLIC_PATHS = {
    "/remote-login", "/api/remote/login", "/manifest.webmanifest", "/sw.js", "/offline.html",
    "/apple-touch-icon.png", "/favicon.ico",
    "/static/remote.css", "/static/remote_login.js", "/static/fonts/Vazirmatn-var.woff2",
}
PUBLIC_PREFIXES = ("/static/icons/",)
# only the PC itself may call these (pairing admin)
LOCAL_ONLY_PATHS = {"/phone", "/static/phone_admin.js"}
LOCAL_ONLY_PREFIXES = ("/api/remote/",)
# never available to a remote client: they act on the PC (not on the conversation)
REMOTE_BLOCKED_PATHS = {
    "/api/shutdown", "/api/open-url", "/api/system-tools/open-folder", "/api/forex/chart",
    "/api/ollama/restart", "/api/settings/models-dir", "/api/settings/browse-models-dir",
    "/api/settings/search", "/api/online/key", "/api/online/key/delete", "/api/voice/reference",
    "/api/models/pull", "/api/models/delete", "/api/diagnostics/mt5",
}
REMOTE_BLOCKED_PREFIXES = ("/api/setup/",)

_LOCK = threading.Lock()
_FAILS = {"count": 0, "locked_until": 0.0}
MAX_FREE_FAILS = 5
MAX_LOCK_SECONDS = 15 * 60


# ---------------------------------------------------------------------------------------------------
# state file
# ---------------------------------------------------------------------------------------------------
def state_path():
    return app_settings.app_data_dir() / "remote_access.json"


def _load():
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def _save(data):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_code():
    return "-".join(
        "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_GROUP_LEN)) for _ in range(CODE_GROUPS)
    )


def _normalize_code(value):
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def pairing_code():
    """The current pairing code; created on first use. Shown only on the PC."""
    with _LOCK:
        data = _load()
        code = data.get("code")
        if not isinstance(code, str) or len(_normalize_code(code)) != CODE_GROUPS * CODE_GROUP_LEN:
            code = generate_code()
            data["code"] = code
            _save(data)
        return code


def regenerate_code(revoke_devices=True):
    with _LOCK:
        data = _load()
        data["code"] = generate_code()
        if revoke_devices:
            data["devices"] = {}
        _save(data)
        return data["code"]


# ---------------------------------------------------------------------------------------------------
# enabled / bind address
# ---------------------------------------------------------------------------------------------------
def is_enabled():
    env = os.environ.get("LOCALCHAT_REMOTE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    remote = app_settings.load_settings().get("remote")
    return bool(isinstance(remote, dict) and remote.get("enabled") is True)


def set_enabled(value):
    settings = app_settings.load_settings()
    remote = settings.get("remote") if isinstance(settings.get("remote"), dict) else {}
    remote["enabled"] = bool(value)
    settings["remote"] = remote
    app_settings._save_settings(settings)


def bind_host():
    override = os.environ.get("LOCALCHAT_BIND", "").strip()
    if override:
        return override
    return "0.0.0.0" if is_enabled() else "127.0.0.1"


def lan_addresses():
    """Private addresses of this PC, best guess first (used to tell the user what to type on the phone)."""
    found = []

    def add(value):
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return
        if ip.version == 4 and not ip.is_loopback and not ip.is_link_local and value not in found:
            found.append(value)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))       # no packet is sent; the OS just picks the outgoing interface
            add(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return found


# ---------------------------------------------------------------------------------------------------
# request classification
# ---------------------------------------------------------------------------------------------------
def _host_of(value):
    try:
        return (urlsplit("//" + (value or "").strip()).hostname or "").lower()
    except ValueError:
        return ""


def _is_loopback_ip(value):
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return ip.is_loopback


def classify(client_ip, headers):
    """'local' only when the request genuinely comes from this PC's own browser/app window."""
    lowered = {str(k).lower() for k in headers.keys()}
    if any(name in lowered for name in PROXY_HEADERS):
        return "remote"
    if not _is_loopback_ip(client_ip):
        return "remote"
    if _host_of(headers.get("Host")) not in LOCAL_HOSTNAMES:
        return "remote"
    return "local"


def host_allowed_for_remote(host):
    """Reject Host headers that are neither a private/CGNAT address nor a LAN / tailnet style name."""
    if not host:
        return False
    extra = {h.strip().lower() for h in os.environ.get("LOCALCHAT_ALLOWED_HOSTS", "").split(",") if h.strip()}
    if host in extra:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return not ip.is_global            # 192.168.x, 10.x, 172.16-31.x, 100.64/10 (Tailscale), fe80::, fd00::
    except ValueError:
        pass
    if "." not in host:                    # bare machine name (http://my-pc:8000)
        return True
    return host.endswith(HOST_SUFFIXES)


def is_public_path(path):
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def is_local_only_path(path):
    return path in LOCAL_ONLY_PATHS or path.startswith(LOCAL_ONLY_PREFIXES)


def is_remote_blocked(path, mutating=True):
    if path in REMOTE_BLOCKED_PATHS:
        return True
    # the Setup Center only READS status from the phone UI; installs / downloads / path changes are POSTs
    return mutating and path.startswith(REMOTE_BLOCKED_PREFIXES)


# ---------------------------------------------------------------------------------------------------
# login / device cookies
# ---------------------------------------------------------------------------------------------------
def lock_remaining():
    with _LOCK:
        return max(0, int(_FAILS["locked_until"] - time.time() + 0.999))


def _register_failure():
    with _LOCK:
        _FAILS["count"] += 1
        extra = _FAILS["count"] - MAX_FREE_FAILS
        if extra >= 0:
            _FAILS["locked_until"] = time.time() + min(MAX_LOCK_SECONDS, 30 * (2 ** extra))


def _register_success():
    with _LOCK:
        _FAILS["count"] = 0
        _FAILS["locked_until"] = 0.0


def reset_rate_limit():
    _register_success()


def try_login(code, device_name="", user_agent=""):
    """Returns (status, payload). status: 'ok' (payload=token) | 'locked' (payload=seconds) | 'bad'."""
    wait = lock_remaining()
    if wait:
        return "locked", wait
    expected = _normalize_code(pairing_code())
    if hmac.compare_digest(_normalize_code(code), expected):
        _register_success()
        return "ok", _issue_device(device_name, user_agent)
    _register_failure()
    wait = lock_remaining()
    return ("locked", wait) if wait else ("bad", 0)


def _issue_device(name, user_agent):
    token = secrets.token_urlsafe(32)
    label = " ".join(str(name or "").split())[:40] or _guess_device(user_agent)
    with _LOCK:
        data = _load()
        devices = data.get("devices") if isinstance(data.get("devices"), dict) else {}
        while len(devices) >= MAX_DEVICES:           # drop the least recently used
            oldest = min(devices, key=lambda k: devices[k].get("last_seen", 0))
            devices.pop(oldest, None)
        devices[_hash(token)] = {"name": label, "created": int(time.time()), "last_seen": int(time.time())}
        data["devices"] = devices
        _save(data)
    return token


def _guess_device(user_agent):
    ua = str(user_agent or "")
    if "iPhone" in ua:
        return "iPhone"
    if "iPad" in ua:
        return "iPad"
    if "Android" in ua:
        return "Android"
    return "Device"


def cookie_value(cookie_header, name=DEVICE_COOKIE):
    for part in (cookie_header or "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value
    return ""


_TOUCH_INTERVAL = 600
_LAST_TOUCH = {}


def device_valid(cookie_header):
    token = cookie_value(cookie_header)
    if not token or len(token) > 200:
        return False
    digest = _hash(token)
    data = _load()
    devices = data.get("devices") if isinstance(data.get("devices"), dict) else {}
    info = devices.get(digest)
    if not isinstance(info, dict):
        return False
    now = time.time()
    if now - _LAST_TOUCH.get(digest, 0) > _TOUCH_INTERVAL:     # keep «last seen» without writing on every request
        _LAST_TOUCH[digest] = now
        with _LOCK:
            fresh = _load()
            entry = (fresh.get("devices") or {}).get(digest)
            if isinstance(entry, dict):
                entry["last_seen"] = int(now)
                _save(fresh)
    return True


def list_devices():
    data = _load()
    devices = data.get("devices") if isinstance(data.get("devices"), dict) else {}
    rows = [{"id": key[:16], "name": v.get("name", ""), "created": v.get("created", 0),
             "last_seen": v.get("last_seen", 0)} for key, v in devices.items() if isinstance(v, dict)]
    return sorted(rows, key=lambda r: r["last_seen"], reverse=True)


def revoke_device(device_id):
    device_id = str(device_id or "")
    if len(device_id) < 8:
        return False
    with _LOCK:
        data = _load()
        devices = data.get("devices") if isinstance(data.get("devices"), dict) else {}
        victims = [k for k in devices if k.startswith(device_id)]
        for key in victims:
            devices.pop(key, None)
        data["devices"] = devices
        _save(data)
    return bool(victims)


def device_cookie_header(token, secure):
    parts = [f"{DEVICE_COOKIE}={token}", "Path=/", "HttpOnly", "SameSite=Strict", f"Max-Age={DEVICE_COOKIE_MAX_AGE}"]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


# ---------------------------------------------------------------------------------------------------
# the single decision function used by the HTTP handler
# ---------------------------------------------------------------------------------------------------
def evaluate(client_ip, headers, path, method, mutating):
    """Decide what to do with a request.

    Returns dict(scope, action, status, message) with action in:
      'pass-local'  – a request from this PC: server.py applies its existing loopback checks;
      'allow'       – an authenticated (or public) remote request;
      'login'       – remote, not signed in: send HTML navigations to /remote-login, others get 401;
      'deny'        – refuse with ``status``/``message``.
    """
    scope = classify(client_ip, headers)
    if scope == "local":
        return {"scope": "local", "action": "pass-local", "status": 200, "message": ""}
    if not is_enabled():
        return {"scope": "remote", "action": "deny", "status": 403,
                "message": "دسترسی از گوشی خاموش است. برنامه را با «start_phone.bat» (یا LOCALCHAT_REMOTE=1) اجرا کنید."}
    host = _host_of(headers.get("Host"))
    if not host_allowed_for_remote(host):
        return {"scope": "remote", "action": "deny", "status": 403, "message": "نام میزبان مجاز نیست."}
    if mutating:
        origin = headers.get("Origin")
        if origin is not None and _host_of(urlsplit(origin).netloc) != host:
            return {"scope": "remote", "action": "deny", "status": 403, "message": "درخواست از مبدأ غیرمجاز رد شد."}
        if (headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
            return {"scope": "remote", "action": "deny", "status": 403, "message": "درخواست از مبدأ غیرمجاز رد شد."}
    if is_local_only_path(path) and path != "/api/remote/login":
        return {"scope": "remote", "action": "deny", "status": 403,
                "message": "این بخش فقط روی خود رایانه در دسترس است."}
    if is_public_path(path):
        return {"scope": "remote", "action": "allow", "status": 200, "message": ""}
    if not device_valid(headers.get("Cookie")):
        return {"scope": "remote", "action": "login", "status": 401, "message": "ابتدا وارد شوید."}
    if is_remote_blocked(path, mutating):
        return {"scope": "remote", "action": "deny", "status": 403,
                "message": "این عملیات روی خود رایانه انجام می‌شود و از گوشی در دسترس نیست."}
    return {"scope": "remote", "action": "allow", "status": 200, "message": ""}


def is_secure_request(headers, is_tls):
    return bool(is_tls or (headers.get("X-Forwarded-Proto") or "").lower() == "https")
