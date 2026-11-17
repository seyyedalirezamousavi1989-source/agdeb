"""Adaptive context: Context Estimator, Context Allocator and the reload policy.

Everything here is a *pure function of its inputs* (model profile, hardware snapshot, runtime state, user ceiling) so
it is unit-testable without Ollama or a GPU. The server only gathers the inputs and applies the result.

Memory model (an estimate – every figure that comes out of it is labelled estimated, never measured)::

    weights  = model file size
    extra(c) = (KV cache(c) · parallel · safety + recurrent state · parallel + compute buffers) · calibration
    need(c)  = weights + extra(c)

Ollama places as much of ``need`` on the GPU as fits and the rest in system RAM, so for a candidate window ``c``::

    gpu_bytes = min(gpu_capacity, need(c))        cpu_bytes = need(c) - gpu_bytes
    feasible  = cpu_bytes <= ram_budget
    slowdown  = read_bytes · (cpu_share + (1 - cpu_share) / GPU_SPEEDUP), relative to the same at the smallest window
                (read_bytes = weights + KV_FILL · KV cache: what decoding reads per token)

The chosen window is the largest candidate that is feasible AND whose estimated slowdown stays under
``SLOWDOWN_TOLERANCE`` – a bigger window that pushes half the model onto the CPU is *possible* but makes every
reply several times slower, so it is not chosen automatically. ``GPU_SPEEDUP`` and the tolerance are planning
parameters (documented below), not measurements.

Stability: the plan must not depend on whether the model happens to be loaded right now (otherwise the window flaps
between requests and Ollama reloads the model every time). Memory used by *Ollama's own* resident models is therefore
counted as reclaimable, and only non-Ollama usage is subtracted from what is free.
"""
import os
import threading
import time
from dataclasses import dataclass, field

try:                                   # the app runs both as a package and as plain modules
    from . import context_arch, hardware
except ImportError:                    # pragma: no cover - script-style launch
    import context_arch
    import hardware

_GIB = 1024 ** 3
VRAM_USABLE = hardware.VRAM_USABLE          # keep headroom for the desktop, CUDA context and fragmentation
OVERHEAD_BYTES = hardware.OVERHEAD_BYTES    # compute buffers Ollama allocates besides weights and KV cache
KV_SAFETY = hardware.KV_SAFETY              # Ollama's real KV allocation sits a little above the textbook formula
GPU_SPEEDUP = 8.0                           # planning assumption: GPU vs CPU memory bandwidth for the bytes read per token
SLOWDOWN_TOLERANCE = 1.25                   # accept up to ~25 % slower decoding than at the smallest window
KV_FILL = 0.5                               # average fraction of a window that is actually filled while decoding
MIN_PLAN_CTX = 4096                         # smallest window we plan for (a model that declares less keeps its own limit)
GROW_AT = 0.6                               # a resident model is reloaded larger once a request needs > 60 % of its window
MAX_CALIBRATION = 4.0
PENDING_TTL = 90.0                          # a requested-but-not-yet-visible load blocks conflicting requests this long


def ram_reserve(total_bytes):
    """RAM kept free for the OS and other programs: 15 % of the machine, at least 2 GiB (scales with the machine)."""
    return max(2 * _GIB, int(0.15 * int(total_bytes or 0)))


def window_steps(upper, lowest=2048):
    """Candidate windows: 2^k and 1.5·2^k up to ``upper`` (plus ``upper`` itself, e.g. a model's odd declared limit)."""
    steps, base = set(), lowest
    while base <= upper:
        steps.add(base)
        if base * 3 // 2 <= upper:
            steps.add(base * 3 // 2)
        base *= 2
    steps.add(int(upper))
    return sorted(steps)


def env_cache_settings(environ=None):
    """KV cache type / parallel slots Ollama is *probably* using, read from LocalChat's own environment.

    Ollama's server may have been started elsewhere with different variables, so these are assumptions and are
    reported as such; the calibration from ``/api/ps`` corrects a wrong guess upwards.
    """
    environ = os.environ if environ is None else environ
    flash = str(environ.get("OLLAMA_FLASH_ATTENTION", "")).strip().lower() in ("1", "true", "yes", "on")
    kv_type = str(environ.get("OLLAMA_KV_CACHE_TYPE", "")).strip().lower()
    kv_type = kv_type if (flash and kv_type in context_arch.KV_ELEMENT_BYTES) else context_arch.DEFAULT_KV_TYPE
    try:
        parallel = max(1, int(environ.get("OLLAMA_NUM_PARALLEL", "1")))
    except ValueError:
        parallel = 1
    return {"kv_type": kv_type, "parallel": parallel}


# ------------------------------------------------------------------------------------------------ estimator
@dataclass(frozen=True)
class ContextCost:
    ctx: int
    kv_bytes: int
    state_bytes: int
    overhead_bytes: int
    weights_bytes: int
    confidence: str          # estimated | upper_bound | fallback_observed | fallback_heuristic
    method: str
    calibration: float = 1.0
    notes: tuple = ()

    @property
    def extra_bytes(self):
        return self.kv_bytes + self.state_bytes + self.overhead_bytes

    @property
    def total_bytes(self):
        return self.weights_bytes + self.extra_bytes


def estimate_cost(profile, ctx, *, kv_type=context_arch.DEFAULT_KV_TYPE, parallel=1, calibration=1.0,
                  legacy_kv_per_token=None, observed=None):
    """Memory the model needs at window ``ctx``.

    Order of trust: architecture adapter (metadata) → a caller-supplied per-token KV figure → what the runtime showed
    for this model (``observed = (ctx, total_bytes)`` from ``/api/ps``) → a size-based heuristic. The last three are
    labelled ``fallback_*``: the allocator then never claims more than it can justify.
    """
    weights = int(profile.file_size or 0)
    cache = context_arch.estimate_cache(profile.arch_meta, ctx, kv_type)
    notes = list(cache.notes)
    if cache.supported:
        kv, state, confidence, method = cache.kv_bytes, cache.state_bytes, cache.confidence, cache.method
    elif legacy_kv_per_token:
        kv, state, confidence, method = int(legacy_kv_per_token) * ctx, 0, "fallback_heuristic", "supplied_kv_per_token"
    elif observed and observed[0] and observed[1] > weights:
        # everything above the weights at the observed window, scaled linearly: includes fixed buffers → upper bound
        per_token = (observed[1] - weights - OVERHEAD_BYTES) / observed[0]
        kv, state, confidence, method = int(max(0.0, per_token) * ctx), 0, "fallback_observed", "observed_runtime"
        notes.append("no architecture formula: cost extrapolated from the size Ollama reported for the loaded model")
    else:
        kv, state = hardware.estimate_kv_bytes_per_token(weights) * ctx, 0
        confidence, method = "fallback_heuristic", "model_size_heuristic"
        notes.append("no architecture formula and no runtime observation: size-based heuristic")
    scaled = lambda value: int(value * calibration)
    return ContextCost(ctx=ctx, kv_bytes=scaled(kv * parallel * KV_SAFETY), state_bytes=scaled(state * parallel),
                       overhead_bytes=scaled(OVERHEAD_BYTES), weights_bytes=weights, confidence=confidence,
                       method=method, calibration=calibration, notes=tuple(notes))


class Calibration:
    """Corrects the estimate upward when Ollama's reported allocation exceeds it (never downward: stay conservative)."""

    def __init__(self):
        self._factors = {}
        self._lock = threading.Lock()

    def factor(self, model):
        with self._lock:
            return self._factors.get(model, 1.0)

    def observe(self, model, profile, ctx, total_bytes, *, kv_type=context_arch.DEFAULT_KV_TYPE, parallel=1,
                legacy_kv_per_token=None):
        """Compare the allocation Ollama reports (``/api/ps`` ``size``) with the prediction for the same window."""
        if not (profile and profile.file_size and ctx and total_bytes and total_bytes > profile.file_size):
            return self.factor(model)
        cost = estimate_cost(profile, ctx, kv_type=kv_type, parallel=parallel, legacy_kv_per_token=legacy_kv_per_token)
        if cost.confidence in ("fallback_heuristic", "fallback_observed"):
            return self.factor(model)        # nothing to compare against that is not itself a guess
        ratio = (total_bytes - profile.file_size) / max(1, cost.extra_bytes)
        with self._lock:
            if ratio > self._factors.get(model, 1.0):
                self._factors[model] = min(MAX_CALIBRATION, ratio)
            return self._factors.get(model, 1.0)


# ------------------------------------------------------------------------------------------------ allocator
@dataclass(frozen=True)
class ContextPlan:
    target: int                  # window to ask Ollama for (largest memory-feasible, performance-acceptable)
    max_feasible: int            # largest window whose memory fits at all (ignoring the slowdown limit)
    upper: int                   # min(model's declared limit, user ceiling)
    limiting: str                # declared | ceiling | performance | memory | memory_insufficient
    confidence: str              # cost confidence at ``target``
    cost: ContextCost
    gpu_bytes: int
    cpu_bytes: int
    slowdown: float
    gpu_capacity: int
    ram_budget: int
    reason: str                  # Persian, for the UI
    measured: dict = field(default_factory=dict)   # which inputs were measured vs assumed
    notes: tuple = ()

    @property
    def fits_gpu(self):
        return self.cpu_bytes == 0 and self.gpu_capacity > 0

    @property
    def gpu_fraction(self):
        total = self.gpu_bytes + self.cpu_bytes
        return round(self.gpu_bytes / total, 3) if total else None

    def as_window(self):
        """Shape of the older ``hardware.pick_window`` result, so existing callers keep working."""
        mib = 1024 ** 2
        kv = self.cost.kv_bytes
        fraction = self.gpu_fraction or 0.0
        return {"num_ctx": self.target, "fits_gpu": self.fits_gpu, "reason": self.reason, "kv_mb": round(kv / mib),
                "gpu_kv_mb": round(kv * fraction / mib), "ram_kv_mb": round(kv * (1 - fraction) / mib),
                "gpu_weight_fraction": fraction, "limiting": self.limiting, "confidence": self.confidence}

    def public(self):
        return {"target": self.target, "max_feasible": self.max_feasible, "upper": self.upper, "limiting": self.limiting,
                "confidence": self.confidence, "gpu_fraction": self.gpu_fraction, "slowdown": round(self.slowdown, 2),
                "estimated_total_gb": round(self.cost.total_bytes / _GIB, 1),
                "kv_cache_gb": round(self.cost.kv_bytes / _GIB, 2), "state_mb": round(self.cost.state_bytes / 1024 ** 2),
                "method": self.cost.method, "measured": dict(self.measured), "notes": list(self.notes), "reason": self.reason}


_REASONS = {
    "declared": "به ظرفیت اعلام‌شدهٔ خود مدل رسید؛ بیشتر از توان مدل درخواست نمی‌شود.",
    "ceiling": "به سقف انتخابی کاربر رسید؛ حافظهٔ بیشتری هم در دسترس است.",
    "performance": "بزرگ‌ترین پنجره‌ای که هنوز سرعت پاسخ را بیش از حد کم نمی‌کند؛ پنجرهٔ بزرگ‌تر بخش بیشتری از مدل را به CPU می‌برد.",
    "memory": "بزرگ‌ترین پنجره‌ای که با VRAM و RAM آزاد جا می‌شود (برآورد، نه اندازه‌گیری).",
    "memory_insufficient": "حافظهٔ آزاد حتی برای کوچک‌ترین پنجره کم است؛ کوچک‌ترین پنجره انتخاب شد و ممکن است اجرا کند یا ناپایدار باشد.",
}


def plan_context(profile, hw, runtime, *, ceiling=None, kv_type=context_arch.DEFAULT_KV_TYPE, parallel=1,
                 calibration=1.0, legacy_kv_per_token=None, declared_override=None, slowdown_tolerance=SLOWDOWN_TOLERANCE,
                 min_ctx=MIN_PLAN_CTX):
    """Largest safe window for ``profile`` on this machine, or None when there is nothing to plan from.

    ``hw`` is a ``hardware.snapshot()`` dict, ``runtime`` a ``runtime_inspector.RuntimeState``. Returns None (so the caller
    keeps its conservative rule) when the model size is unknown, or when neither a measured GPU nor a model that is
    already loaded tells us where the weights will live.
    """
    weights = int(profile.file_size or 0)
    if not weights:
        return None
    ram, gpu = hw.get("ram"), hw.get("gpu")
    loaded = runtime.find(profile.name) if runtime is not None else None
    measured = {"ram": bool(ram), "vram": bool(gpu and gpu.get("used") is not None), "runtime": loaded is not None}
    notes = []

    # --- GPU capacity (stable: Ollama's own resident models are reclaimable, only other programs count) ----------
    if gpu and gpu.get("total"):
        ollama_vram = runtime.total_vram if runtime is not None and runtime.reachable else 0
        other = max(0, int(gpu["used"]) - ollama_vram) if gpu.get("used") is not None else 0
        if gpu.get("used") is None:
            notes.append("GPU memory usage unavailable: other programs' VRAM use is not subtracted")
        gpu_capacity = max(0, int(gpu["total"] * VRAM_USABLE) - other)
    elif loaded is not None and ram:
        # no GPU telemetry, but Ollama says how much of this model it keeps on a GPU: that is all we can count on
        gpu_capacity = loaded.size_vram
        notes.append("GPU capacity taken from Ollama's own report (no GPU telemetry)")
    else:
        return None        # no GPU data and no loaded model + RAM reading: do not guess where Ollama puts the weights

    # --- RAM budget (stable the same way: Ollama's resident CPU share is reclaimable) ----------------------------
    if ram:
        reclaimable = runtime.total_cpu if runtime is not None and runtime.reachable else 0
        ram_budget = max(0, int(ram["available"]) + reclaimable - ram_reserve(ram["total"]))
    else:
        ram_budget = 0
        notes.append("RAM unavailable: only a plan that fits entirely on the GPU is accepted")

    declared = declared_override or profile.declared_context
    if declared is None:
        notes.append("model declares no context limit")
    upper = min(int(ceiling) if ceiling else 10 ** 9, int(declared) if declared else 10 ** 9)
    if upper == 10 ** 9:
        upper = 8192
        notes.append("no declared limit and no user ceiling: limited to 8192")
    floor = min(min_ctx, upper)
    observed = (loaded.context_length, loaded.size) if loaded is not None and loaded.context_length and loaded.size else None
    cost_of = lambda c: estimate_cost(profile, c, kv_type=kv_type, parallel=parallel, calibration=calibration,
                                      legacy_kv_per_token=legacy_kv_per_token, observed=observed)

    def evaluate(ctx):
        cost = cost_of(ctx)
        gpu_bytes = min(gpu_capacity, cost.total_bytes)
        cpu_bytes = cost.total_bytes - gpu_bytes
        feasible = cpu_bytes <= ram_budget if ram else cpu_bytes == 0
        # Decoding reads the weights plus the filled part of the KV cache for every token. A window is assumed to be
        # about half full on average (KV_FILL); bytes on the CPU are read ~GPU_SPEEDUP times slower than on the GPU.
        read_bytes = cost.weights_bytes + KV_FILL * cost.kv_bytes
        cpu_share = cpu_bytes / cost.total_bytes if cost.total_bytes else 0.0
        return cost, gpu_bytes, cpu_bytes, feasible, read_bytes * (cpu_share + (1 - cpu_share) / GPU_SPEEDUP)

    base_cost, base_gpu, base_cpu, base_ok, base_time = evaluate(floor)
    chosen = max_feasible = None
    results = {}
    for ctx in window_steps(upper, lowest=min(2048, floor)):
        if ctx < floor:
            continue
        cost, gpu_bytes, cpu_bytes, feasible, time_cost = evaluate(ctx)
        # entirely GPU-resident windows are never penalised (their KV reads are cheap); a spill to the CPU is
        slowdown = 1.0 if cpu_bytes == 0 else (time_cost / base_time if base_time else 1.0)
        results[ctx] = (cost, gpu_bytes, cpu_bytes, slowdown)
        if feasible:
            max_feasible = ctx
            if slowdown <= slowdown_tolerance:
                chosen = ctx
    if not base_ok or chosen is None:
        chosen, limiting = floor, "memory_insufficient"
        max_feasible = max_feasible or floor
        results.setdefault(floor, (base_cost, base_gpu, base_cpu, 1.0))
    elif chosen == upper:
        limiting = "declared" if declared and upper == declared and (not ceiling or declared <= ceiling) else "ceiling"
    elif max_feasible is not None and chosen < max_feasible:
        limiting = "performance"
    else:
        limiting = "memory"
    cost, gpu_bytes, cpu_bytes, slowdown = results[chosen]
    if cost.confidence.startswith("fallback"):
        notes.append("memory cost is a fallback estimate (no usable architecture metadata)")
    return ContextPlan(target=chosen, max_feasible=max_feasible or chosen, upper=upper, limiting=limiting,
                       confidence=cost.confidence, cost=cost, gpu_bytes=gpu_bytes, cpu_bytes=cpu_bytes,
                       slowdown=slowdown, gpu_capacity=gpu_capacity, ram_budget=ram_budget,
                       reason=_REASONS[limiting], measured=measured, notes=tuple(notes + list(cost.notes)))


# ------------------------------------------------------------------------------------------------ reload policy
@dataclass(frozen=True)
class Decision:
    num_ctx: int
    action: str        # cold_load | keep | grow | shrink | follow_pending
    reload: bool       # True when applying num_ctx makes Ollama reload a model that is resident with another window
    reason: str


class ContextCoordinator:
    """Decides which ``num_ctx`` to send so the model is reloaded only when that is worth it, and never twice at once.

    * cold model        → load at the planned target (no reload exists to avoid);
    * resident, fits    → keep the resident window (a reload costs seconds to minutes);
    * resident, too small for the request (> ``GROW_AT`` of the window) → grow straight to the target, once;
    * resident, above what memory/ceiling allow → shrink to the target;
    * load already requested but not yet visible in ``/api/ps`` → follow it (no conflicting concurrent window).
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._pending = {}
        self._lock = threading.Lock()

    def decide(self, model, plan, actual, required_tokens=0):
        key = str(model).lower().removesuffix(":latest")
        now = self._clock()
        with self._lock:
            pending = self._pending.get(key)
            if actual is not None:
                self._pending.pop(key, None)
            elif pending and now - pending[1] < PENDING_TTL:
                return Decision(pending[0], "follow_pending", False, "a load at this window is already in progress")
            if actual is None:
                decision = Decision(plan.target, "cold_load", False, "model not loaded: loading at the planned window")
            elif actual == plan.target:
                decision = Decision(actual, "keep", False, "resident window equals the plan")
            elif actual > plan.upper or actual > plan.max_feasible:
                decision = Decision(plan.target, "shrink", True,
                                    "resident window exceeds the model limit, the user ceiling or free memory")
            elif actual < plan.target and required_tokens > GROW_AT * actual:
                decision = Decision(plan.target, "grow", True, "request no longer fits comfortably in the resident window")
            else:
                decision = Decision(actual, "keep", False, "keeping the resident window to avoid a reload")
            if decision.action in ("cold_load", "grow", "shrink"):
                self._pending[key] = (decision.num_ctx, now)
            return decision

    def clear(self, model=None):
        with self._lock:
            if model is None:
                self._pending.clear()
            else:
                self._pending.pop(str(model).lower().removesuffix(":latest"), None)
