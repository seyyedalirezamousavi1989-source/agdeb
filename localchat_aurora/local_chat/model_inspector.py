"""Model Inspector: turn Ollama's ``/api/show`` (+ the ``/api/tags`` entry) into one validated ``ModelProfile``.

Nothing here talks to the network: ``parse_show`` is a pure function over the JSON, so every shape of metadata
(complete, partial, malformed, foreign architecture) is unit-testable. ``ModelInspector`` adds a small cache because
model metadata is stable until the model file changes, so it costs one ``/api/show`` call per model per
``STABLE_TTL``, not one per request (``invalidate`` forces a refresh, e.g. after a model pull).

The profile keeps three things apart that used to be mixed up:

* ``declared_context`` – what the model *file* says it supports (metadata, not proof of anything running),
* ``modelfile_num_ctx`` – a default ``num_ctx`` baked into the Modelfile's ``PARAMETER`` lines (may be far smaller),
* the runtime state, which lives in :mod:`runtime_inspector` and is never read from here.
"""
import re
import threading
import time
from dataclasses import dataclass, field

_ARCH_KEYS = ("block_count", "context_length", "embedding_length", "feed_forward_length", "full_attention_interval",
              "nextn_predict_layers", "attention.head_count", "attention.head_count_kv", "attention.key_length",
              "attention.value_length", "attention.sliding_window", "attention.kv_lora_rank",
              "ssm.conv_kernel", "ssm.group_count", "ssm.inner_size", "ssm.state_size", "ssm.time_step_rank",
              "rope.dimension_count", "rope.freq_base")
_SIZE_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
STABLE_TTL = 600.0           # metadata only changes when the model file does; re-read every 10 minutes
FAILURE_TTL = 20.0           # a failed lookup (Ollama busy / down) is retried soon, but not on every request


@dataclass(frozen=True)
class ModelProfile:
    name: str
    architecture: str = None
    family: str = None
    parameter_count: int = None
    quantization: str = None
    file_size: int = None
    declared_context: int = None
    modelfile_num_ctx: int = None
    capabilities: tuple = ()
    arch_meta: dict = field(default_factory=dict)      # architecture keys with the «<arch>.» prefix removed
    issues: tuple = ()                                  # what was missing or malformed (never silently ignored)
    digest: str = None

    @property
    def hybrid(self):
        return bool(self.arch_meta.get("full_attention_interval")) or any(k.startswith("ssm.") for k in self.arch_meta)

    @property
    def complete(self):
        return not self.issues

    def public(self):
        return {"name": self.name, "architecture": self.architecture, "family": self.family,
                "parameter_count": self.parameter_count, "quantization": self.quantization,
                "file_size": self.file_size, "declared_context": self.declared_context,
                "modelfile_num_ctx": self.modelfile_num_ctx, "hybrid": self.hybrid,
                "capabilities": list(self.capabilities), "issues": list(self.issues)}


def parse_parameter_size(text):
    """«27.3B» → 27_300_000_000 (None when not a recognisable size)."""
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT])\s*", str(text or ""), re.I)
    return int(float(match.group(1)) * _SIZE_SUFFIX[match.group(2).upper()]) if match else None


def _positive_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value == int(value):
        return int(value)
    return None


def _modelfile_num_ctx(parameters):
    """``num_ctx`` from the Modelfile ``parameters`` text of /api/show (e.g. «num_ctx   8192»), else None."""
    if not isinstance(parameters, str):
        return None
    match = re.search(r"^\s*num_ctx\s+(\d+)\s*$", parameters, re.M)
    return _positive_int(int(match.group(1))) if match else None


def parse_show(name, show, tags_entry=None):
    """Validated :class:`ModelProfile` from ``/api/show`` JSON (``show``) and the ``/api/tags`` entry (size/digest)."""
    issues = []
    show = show if isinstance(show, dict) else {}
    if not show:
        issues.append("/api/show returned no usable data")
    info = show.get("model_info")
    if info is not None and not isinstance(info, dict):
        issues.append("model_info is not an object")
    info = info if isinstance(info, dict) else {}
    details = show.get("details") if isinstance(show.get("details"), dict) else {}
    tags_entry = tags_entry if isinstance(tags_entry, dict) else {}

    arch = info.get("general.architecture")
    arch = arch if isinstance(arch, str) and arch.strip() else None
    if arch is None:
        arch = details.get("family") if isinstance(details.get("family"), str) else None
        issues.append("general.architecture missing")

    meta = {}
    if arch:
        prefix = arch + "."
        for key in _ARCH_KEYS:
            value = info.get(prefix + key)
            if value is None:
                continue
            if isinstance(value, bool) or not (isinstance(value, (int, float, list))):
                issues.append(f"{prefix}{key} has an unsupported type")
                continue
            if isinstance(value, list) and not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                issues.append(f"{prefix}{key} is not a numeric list")
                continue
            meta[key] = value
    # Prefer the architecture's own context_length; the old code took the minimum over every «*.context_length»
    # key, which also matched unrelated sub-model keys.
    declared = _positive_int(meta.get("context_length"))
    if declared is None:
        others = [v for k, v in info.items() if str(k).endswith(".context_length") and _positive_int(v)]
        declared = min(others) if others else None
        issues.append("declared context length missing" if declared is None
                      else "architecture context_length missing: smallest context_length key used")
    for required in ("block_count", "attention.head_count"):
        if required not in meta:
            issues.append(f"{required} missing")

    parameter_count = _positive_int(info.get("general.parameter_count")) or parse_parameter_size(details.get("parameter_size"))
    quant = details.get("quantization_level")
    caps = show.get("capabilities")
    return ModelProfile(
        name=str(name), architecture=arch,
        family=details.get("family") if isinstance(details.get("family"), str) else arch,
        parameter_count=parameter_count,
        quantization=quant if isinstance(quant, str) and quant else None,
        file_size=_positive_int(tags_entry.get("size")), declared_context=declared,
        modelfile_num_ctx=_modelfile_num_ctx(show.get("parameters")),
        capabilities=tuple(c for c in caps if isinstance(c, str)) if isinstance(caps, list) else (),
        arch_meta=meta, issues=tuple(dict.fromkeys(issues)),
        digest=tags_entry.get("digest") if isinstance(tags_entry.get("digest"), str) else None)


class ModelInspector:
    """Caching front end. ``fetch_show(name)`` / ``fetch_tags()`` are injected so tests need no Ollama."""

    def __init__(self, fetch_show, fetch_tags, clock=time.monotonic):
        self._fetch_show, self._fetch_tags, self._clock = fetch_show, fetch_tags, clock
        self._cache = {}
        self._lock = threading.Lock()

    def inspect(self, name, force=False):
        now = self._clock()
        with self._lock:
            hit = self._cache.get(name)
            if hit and not force and now < hit[0]:
                return hit[1]
        try:
            show = self._fetch_show(name)
            ok = True
        except (OSError, ValueError, TypeError, AttributeError, TimeoutError) as exc:
            show, ok = {}, False
            error = f"/api/show failed: {type(exc).__name__}"
        try:
            tags = next((t for t in (self._fetch_tags() or ()) if isinstance(t, dict)
                         and name in (t.get("name"), t.get("model"))), {})
        except (OSError, ValueError, TypeError, AttributeError, TimeoutError):
            tags = {}
        profile = parse_show(name, show, tags)
        if not ok:
            profile = ModelProfile(**{**profile.__dict__, "issues": (error,) + profile.issues})
        with self._lock:
            self._cache[name] = (now + (STABLE_TTL if ok else FAILURE_TTL), profile)
        return profile

    def invalidate(self, name=None):
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop(name, None)
