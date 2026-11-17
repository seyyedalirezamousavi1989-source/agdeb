"""Runtime Inspector: what Ollama is *actually* running, as opposed to what a model declares or what we asked for.

Three different numbers used to be treated as one:

* ``declared``  – the model file's maximum context (``/api/show``),
* ``requested`` – the ``num_ctx`` LocalChat put in the request options,
* ``actual``    – the ``context_length`` Ollama reports for the loaded model in ``/api/ps``.

``ContextTriple`` keeps them apart and says whether the request is *verified*. A context change is never reported as
successful unless ``/api/ps`` confirms it; when Ollama does not report ``context_length`` (older versions) the status is
``unknown``, never ``verified``.

``fetch_ps`` is injected and must RAISE when Ollama cannot be reached, so «Ollama is down» is not confused with
«no model is loaded» (the old helper returned ``[]`` for both).
"""
import threading
import time
from dataclasses import dataclass

PS_TTL = 2.0          # several callers inside one chat turn share one /api/ps request


def same_model(a, b):
    norm = lambda n: str(n or "").strip().lower().removesuffix(":latest")
    return norm(a) == norm(b)


def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size: int = 0                 # total bytes Ollama holds for this model (weights + cache + buffers)
    size_vram: int = 0            # the part that sits in GPU memory
    context_length: int = None    # None = Ollama did not report it
    expires_at: str = None

    @property
    def size_cpu(self):
        return max(0, self.size - self.size_vram)

    @property
    def gpu_fraction(self):
        return (self.size_vram / self.size) if self.size > 0 else None


def parse_ps(payload):
    """``/api/ps`` JSON → ``[LoadedModel]``; malformed entries are skipped, never raised on."""
    models = payload.get("models") if isinstance(payload, dict) else None
    result = []
    for item in models if isinstance(models, list) else ():
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        result.append(LoadedModel(name=name, size=_int(item.get("size")) or 0, size_vram=_int(item.get("size_vram")) or 0,
                                  context_length=_int(item.get("context_length")) or None,
                                  expires_at=item.get("expires_at") if isinstance(item.get("expires_at"), str) else None))
    return result


@dataclass(frozen=True)
class RuntimeState:
    reachable: bool
    models: tuple = ()
    error: str = None
    raw: tuple = ()               # the untouched /api/ps entries (other parts of the app read more fields from them)

    def find(self, name):
        return next((m for m in self.models if same_model(m.name, name)), None)

    @property
    def total_vram(self):
        return sum(m.size_vram for m in self.models)

    @property
    def total_cpu(self):
        return sum(m.size_cpu for m in self.models)


@dataclass(frozen=True)
class ContextTriple:
    declared: int = None
    requested: int = None
    actual: int = None
    status: str = "not_loaded"    # verified | mismatch | unknown | not_loaded | unreachable
    detail: str = ""

    @property
    def reload_required(self):
        """True when applying ``requested`` means Ollama has to reload a model that is resident with another window."""
        return self.status == "mismatch"

    def public(self):
        return {"declared": self.declared, "requested": self.requested, "actual": self.actual,
                "status": self.status, "verified": self.status == "verified", "detail": self.detail}


def compare(declared, requested, loaded, reachable=True):
    """Classify one model's context state (pure; see :class:`ContextTriple`)."""
    if not reachable:
        return ContextTriple(declared, requested, None, "unreachable", "Ollama could not be reached")
    if loaded is None:
        return ContextTriple(declared, requested, None, "not_loaded", "model is not loaded")
    if loaded.context_length is None:
        return ContextTriple(declared, requested, None, "unknown", "Ollama did not report context_length")
    if requested is None or loaded.context_length == requested:
        return ContextTriple(declared, requested, loaded.context_length, "verified")
    return ContextTriple(declared, requested, loaded.context_length, "mismatch",
                         f"requested {requested} but Ollama runs {loaded.context_length}")


class RuntimeInspector:
    def __init__(self, fetch_ps, clock=time.monotonic):
        self._fetch_ps, self._clock = fetch_ps, clock
        self._cached = None
        self._at = 0.0
        self._lock = threading.Lock()

    def read(self, fresh=False):
        now = self._clock()
        with self._lock:
            if not fresh and self._cached is not None and now - self._at < PS_TTL:
                return self._cached
        try:
            payload = self._fetch_ps()
            raw = tuple(m for m in (payload.get("models") or ()) if isinstance(m, dict)) if isinstance(payload, dict) else ()
            state = RuntimeState(True, tuple(parse_ps(payload)), raw=raw)
        except (OSError, ValueError, TypeError, AttributeError, TimeoutError) as exc:
            state = RuntimeState(False, (), f"{type(exc).__name__}: {exc}"[:200])
        with self._lock:
            self._cached, self._at = state, now
        return state

    @property
    def last_reachable(self):
        with self._lock:
            return self._cached.reachable if self._cached is not None else True

    def forget(self):
        with self._lock:
            self._cached = None

    def context_state(self, name, declared=None, requested=None, fresh=False):
        state = self.read(fresh=fresh)
        return compare(declared, requested, state.find(name), state.reachable)
