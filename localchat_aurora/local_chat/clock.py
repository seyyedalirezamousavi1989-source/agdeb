"""Internal clock: correct time without the model, without the internet, and without a request per turn.

* ``now()`` = the computer's clock + a saved correction. It never touches the network.
* ``sync()`` reads the HTTP ``Date`` header of a well-known HTTPS site ONCE, stores the correction in the settings
  file and from then on works offline. It is only called on the user's explicit action or when the app is already
  online for the web tab (see ``maybe_sync``) — a local-first app must not phone home on its own.
* The time zone is always the computer's own; only the instant is corrected.
"""
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen

SYNC_HOSTS = ("https://www.cloudflare.com", "https://www.microsoft.com", "https://www.google.com")
RESYNC_AFTER_DAYS = 30
NEGLIGIBLE_SECONDS = 2.0          # below this the PC clock is already right; do not carry a pointless offset
SANE_LIMIT_SECONDS = 3 * 365 * 86400

_LOCK = threading.Lock()
_state = {"offset": 0.0, "synced_at": None, "source": "", "loaded": False}


def _store():
    """(load, save) of the persisted state; imported lazily so the module is usable in isolation."""
    try:
        from . import settings as app_settings
    except ImportError:  # pragma: no cover - flat import when run as scripts
        import settings as app_settings
    return app_settings.clock_state, app_settings.set_clock_state


def _load():
    if _state["loaded"]:
        return
    try:
        saved = _store()[0]()
    except Exception:  # noqa: BLE001 - unreadable settings: fall back to the plain system clock
        saved = {}
    _state.update(offset=float(saved.get("offset", 0.0) or 0.0), synced_at=saved.get("synced_at"),
                  source=str(saved.get("source") or ""), loaded=True)


def reset_for_tests():
    with _LOCK:
        _state.update(offset=0.0, synced_at=None, source="", loaded=False)


def epoch():
    with _LOCK:
        _load()
        return time.time() + _state["offset"]


def now():
    """Corrected current time as an aware datetime in the computer's own time zone."""
    return datetime.fromtimestamp(epoch(), tz=timezone.utc).astimezone()


def status():
    with _LOCK:
        _load()
        synced = _state["synced_at"]
        age_days = round((time.time() - synced) / 86400, 1) if isinstance(synced, (int, float)) else None
        return {"offset_seconds": round(_state["offset"], 1), "synced_at": synced, "age_days": age_days,
                "source": _state["source"], "synced": synced is not None}


def sync(opener=urlopen, hosts=SYNC_HOSTS, timeout=6):
    """Fetch the true time once and store the correction. Raises OSError when no host answers."""
    last_error = None
    for host in hosts:
        try:
            before = time.time()
            request = Request(host, method="HEAD", headers={"User-Agent": "LocalChat-clock/1"})
            with opener(request, timeout=timeout) as response:
                header = response.headers.get("Date")
            after = time.time()
            if not header:
                continue
            remote = parsedate_to_datetime(header).timestamp()
            offset = remote - (before + after) / 2          # half the round trip is the network delay
            if abs(offset) > SANE_LIMIT_SECONDS:
                continue
            if abs(offset) < NEGLIGIBLE_SECONDS:
                offset = 0.0
            with _LOCK:
                _state.update(offset=float(offset), synced_at=after, source=host.split("//", 1)[-1], loaded=True)
                _store()[1]({"offset": float(offset), "synced_at": after, "source": _state["source"]})
            return status()
        except (OSError, ValueError, TypeError) as exc:
            last_error = exc
    raise OSError(f"دریافت زمان از اینترنت ممکن نشد: {last_error}")


def maybe_sync(opener=urlopen):
    """Called when the app is online anyway (web search): refresh in the background if never synced or stale."""
    info = status()
    if info["synced"] and info["age_days"] is not None and info["age_days"] < RESYNC_AFTER_DAYS:
        return False

    def worker():
        try:
            sync(opener)
        except OSError:
            pass

    threading.Thread(target=worker, name="clock-sync", daemon=True).start()
    return True
