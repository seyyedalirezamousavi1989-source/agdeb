"""Hardware-aware context window.

The old rule was «model bigger than 10 GB → 4096 tokens, otherwise 8192». That wastes a big GPU and starves a
small one. Here the window is the largest step whose KV-cache still fits next to the model weights in VRAM:

    weights (model file size) + compute buffers + KV-cache(window) + other programs' VRAM  <=  90 % of VRAM

Everything degrades safely: no NVIDIA GPU / no ``nvidia-smi`` → ``pick_window`` returns None and the caller keeps its
conservative size-based rule. Nothing here talks to the network.

Hardware Inspector (``snapshot``): RAM, every detectable GPU and the CPU, each value tagged with where it came from
(``source``). A value that could not be measured is ``None`` and listed in ``unavailable`` – never invented. GPU
discovery does not depend on ``nvidia-smi`` alone: NVML (ctypes, no extra package) → ``nvidia-smi`` → the Windows
display-adapter registry/performance counters (any vendor) → Linux ``/sys/class/drm`` (AMD). Integrated GPUs that share
system RAM are reported with ``dedicated=False`` and are never treated as VRAM. ``pick_window`` is the older
VRAM/RAM planner and is kept for compatibility; the adaptive allocator in ``adaptive_context`` is what the server uses.
"""
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time

_MIB = 1024 ** 2
_GB = 1024 ** 3
STEPS = (4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536)
DEFAULT_MAX_CONTEXT = 32768       # longer windows make every prompt slower; the user can raise it in the memory panel
VRAM_USABLE = 0.90                # keep headroom for the desktop, CUDA context and fragmentation
OVERHEAD_BYTES = int(0.7 * _GB)   # compute / scratch buffers Ollama allocates besides weights and KV-cache
KV_SAFETY = 1.15                  # Ollama's real KV allocation is a bit above the textbook formula
_CACHE = {"at": 0.0, "value": None}
_LOCK = threading.Lock()


def _nvidia_smi_path():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.isfile(candidate):
            return candidate
    return None


def _query_gpus(run=subprocess.run):
    exe = _nvidia_smi_path()
    if not exe:
        return []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = run([exe, "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=4, creationflags=flags)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if getattr(out, "returncode", 1) != 0:
        return []
    gpus = []
    for line in (out.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total, used, free = (int(float(p)) * _MIB for p in parts[-3:])
        except ValueError:
            continue
        gpus.append({"name": ", ".join(parts[:-3]) or "GPU", "total": total, "used": used, "free": free})
    return gpus


def all_gpus(ttl=20, run=subprocess.run):
    """Every detectable GPU (see the module docstring for the discovery order). Cached for ``ttl`` seconds."""
    now = time.monotonic()
    with _LOCK:
        if _CACHE["value"] is not None and now - _CACHE["at"] < ttl:
            return list(_CACHE["value"])
    gpus = _query_nvml()
    if not gpus:
        gpus = [_gpu_entry(g["name"], g["total"], g["used"], "nvidia-smi", dedicated=True) for g in _query_gpus(run)]
    if not gpus:
        gpus = _query_windows_gpus(run) or _query_sysfs_gpus()
    with _LOCK:
        _CACHE.update(at=now, value=list(gpus))
    return gpus


def gpu_info(ttl=20, run=subprocess.run):
    """The dedicated GPU Ollama will most likely use (the one with the most memory), or None.

    ``used``/``free`` are None when the platform only exposes the adapter's size (never a made-up number).
    """
    dedicated = [g for g in all_gpus(ttl, run) if g.get("dedicated")]
    return max(dedicated, key=lambda g: g["total"]) if dedicated else None


def _ram_fallback():
    """RAM without psutil: Windows GlobalMemoryStatusEx, or Linux /proc/meminfo. None when neither works."""
    try:
        if os.name == "nt":
            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {"total": int(status.ullTotalPhys), "available": int(status.ullAvailPhys),
                        "used": int(status.ullTotalPhys - status.ullAvailPhys), "source": "GlobalMemoryStatusEx"}
        elif os.path.exists("/proc/meminfo"):
            values = {}
            with open("/proc/meminfo", encoding="ascii", errors="replace") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    number = rest.split()
                    if number and number[0].isdigit():
                        values[key] = int(number[0]) * 1024
            if values.get("MemTotal") and "MemAvailable" in values:
                return {"total": values["MemTotal"], "available": values["MemAvailable"],
                        "used": values["MemTotal"] - values["MemAvailable"], "source": "/proc/meminfo"}
    except Exception:  # noqa: BLE001 - telemetry must never break chat
        pass
    return None


def ram_info():
    """``{"total", "available", "used", "source"}`` in bytes, or None when RAM cannot be measured."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {"total": int(memory.total), "available": int(memory.available),
                "used": int(memory.total - memory.available), "source": "psutil"}
    except Exception:  # noqa: BLE001 - psutil missing or failing must not break chat
        return _ram_fallback()


# ---------------------------------------------------------------------------------------- GPU discovery
_INTEGRATED = re.compile(r"\b(?:UHD|Iris|HD Graphics|Radeon\(TM\) Graphics|Radeon Graphics|Vega \d+ Graphics|Intel\(R\) Graphics)\b", re.I)


def _vendor(name):
    lowered = str(name).lower()
    for vendor, needles in (("nvidia", ("nvidia", "geforce", "quadro", "rtx", "tesla")), ("amd", ("amd", "radeon")),
                            ("intel", ("intel", "arc "))):
        if any(n in lowered for n in needles):
            return vendor
    return "unknown"


def _gpu_entry(name, total, used, source, dedicated=None):
    name = str(name or "GPU").strip() or "GPU"
    if dedicated is None:
        dedicated = not _INTEGRATED.search(name)
    free = (max(0, total - used) if isinstance(total, int) and isinstance(used, int) else None)
    return {"name": name, "vendor": _vendor(name), "total": total, "used": used, "free": free,
            "dedicated": bool(dedicated), "source": source}


def _query_nvml(lib=None):
    """NVIDIA GPUs through NVML (ships with the driver; no pynvml needed). [] when NVML is unavailable."""
    try:
        if lib is None:
            names = (["nvml.dll", r"C:\Windows\System32\nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
                     if os.name == "nt" else ["libnvidia-ml.so.1", "libnvidia-ml.so"])
            for candidate in names:
                try:
                    lib = ctypes.CDLL(candidate)
                    break
                except OSError:
                    continue
            if lib is None:
                return []

        class _Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        if lib.nvmlInit_v2() != 0:
            return []
        try:
            count = ctypes.c_uint(0)
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                return []
            gpus = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                memory, name = _Memory(), ctypes.create_string_buffer(96)
                if (lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0
                        or lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0):
                    continue
                label = (name.value.decode("utf-8", "replace") if lib.nvmlDeviceGetName(handle, name, 96) == 0 else "NVIDIA GPU")
                gpus.append(_gpu_entry(label, int(memory.total), int(memory.used), "nvml", dedicated=True))
            return gpus
        finally:
            lib.nvmlShutdown()
    except Exception:  # noqa: BLE001 - a broken driver library must not break chat
        return []


_WIN_GPU_SCRIPT = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$cls='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}';"
    "$a=@(Get-ChildItem $cls | ForEach-Object { $p=Get-ItemProperty $_.PSPath;"
    " if($p.DriverDesc -and $p.'HardwareInformation.qwMemorySize'){ @{name=[string]$p.DriverDesc;total=[int64]$p.'HardwareInformation.qwMemorySize'} } });"
    "$u=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum;"
    "@{adapters=$a;dedicated_usage=$(if($u.Count -gt 0){[int64]$u.Sum}else{$null})} | ConvertTo-Json -Compress -Depth 4")


def _parse_windows_gpus(text):
    """Pure parser for the PowerShell output above (display adapters from the registry + total dedicated usage)."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    adapters = data.get("adapters") if isinstance(data, dict) else None
    if isinstance(adapters, dict):                    # PowerShell collapses a one-element array to an object
        adapters = [adapters]
    adapters = [a for a in adapters or () if isinstance(a, dict) and isinstance(a.get("total"), int) and a["total"] > 0]
    usage = data.get("dedicated_usage") if isinstance(data, dict) else None
    dedicated = [a for a in adapters if not _INTEGRATED.search(str(a.get("name", "")))]
    gpus = []
    for adapter in adapters:
        is_dedicated = adapter in dedicated
        # the usage counter is per adapter instance, not per name: attribute it only when it is unambiguous
        used = usage if (is_dedicated and len(dedicated) == 1 and isinstance(usage, int)) else None
        gpus.append(_gpu_entry(adapter.get("name"), adapter["total"], used, "windows-registry", dedicated=is_dedicated))
    return gpus


def _query_windows_gpus(run=subprocess.run):
    if os.name != "nt":
        return []
    try:
        out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_GPU_SCRIPT], capture_output=True,
                  text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return _parse_windows_gpus(out.stdout) if getattr(out, "returncode", 1) == 0 else []


def _query_sysfs_gpus(root="/sys/class/drm"):
    """AMD (and other amdgpu/ROCm) VRAM counters on Linux."""
    gpus = []
    try:
        for card in sorted(os.listdir(root)):
            base = os.path.join(root, card, "device")
            try:
                total = int(open(os.path.join(base, "mem_info_vram_total"), encoding="ascii").read().strip())
                used = int(open(os.path.join(base, "mem_info_vram_used"), encoding="ascii").read().strip())
            except (OSError, ValueError):
                continue
            try:
                name = open(os.path.join(base, "product_name"), encoding="utf-8", errors="replace").read().strip()
            except OSError:
                name = "AMD GPU"
            gpus.append(_gpu_entry(name, total, used, "sysfs", dedicated=True))
    except OSError:
        return []
    return gpus


def kv_bytes_per_token(show_info, kv_type_bytes=2):
    """KV-cache bytes one token costs, from Ollama's ``/api/show`` ``model_info`` (None when it lacks the fields)."""
    info = show_info.get("model_info") if isinstance(show_info, dict) else None
    if not isinstance(info, dict):
        return None
    arch = info.get("general.architecture")
    if not isinstance(arch, str):
        return None

    def number(*suffixes):
        for suffix in suffixes:
            value = info.get(f"{arch}.{suffix}")
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
            if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
                return int(max(value))            # per-layer lists (some hybrid models): be pessimistic
        return None

    layers = number("block_count")
    heads = number("attention.head_count")
    kv_heads = number("attention.head_count_kv") or heads
    key_len = number("attention.key_length")
    value_len = number("attention.value_length") or key_len
    if not key_len:
        width = number("embedding_length")
        key_len = value_len = (width // heads) if width and heads else None
    if not (layers and kv_heads and key_len and value_len):
        return None
    return layers * kv_heads * (key_len + value_len) * kv_type_bytes


def estimate_kv_bytes_per_token(model_bytes):
    """Rough fallback from the model file size alone (8B ≈ 0.125 MB/token, 32B ≈ 0.25 MB/token), pessimistic."""
    return int(max(model_bytes, 1) * 2.8e-5) if model_bytes < 6 * _GB else int(max(model_bytes, 1) * 2.0e-5)


def pick_window(model_bytes, kv_per_token, capacity, gpu, other_vram_bytes=0, max_context=DEFAULT_MAX_CONTEXT,
                min_context=4096, ram=None, model_vram_bytes=None, reserve_ram_bytes=None):
    """Choose the largest safe context using BOTH VRAM and system RAM.

    Ollama commonly offloads large models across GPU and CPU. The previous implementation
    subtracted the *entire* model file from VRAM; if weights exceeded VRAM it immediately
    fell back to 4096 even when substantial system RAM was available. We now estimate the
    GPU/CPU weight split and require the corresponding KV-cache portions to fit each pool.
    This is deliberately conservative and is a planning estimate, not a promise from Ollama.
    """
    if not model_bytes or (not gpu and not ram):
        return None
    kv = int(kv_per_token or estimate_kv_bytes_per_token(model_bytes))
    limit = min(int(max_context or DEFAULT_MAX_CONTEXT), int(capacity) if capacity else 10 ** 9)
    # Never inflate a model's declared context capacity just to reach our preferred minimum.
    if limit < min_context:
        return {"num_ctx": max(1, limit), "fits_gpu": False, "kv_mb": 0,
                "gpu_kv_mb": 0, "ram_kv_mb": 0, "gpu_weight_fraction": 0.0,
                "reason": "ظرفیت اعلام‌شدهٔ خود مدل از حداقل پنجرهٔ معمول کمتر است؛ از ظرفیت واقعی مدل تجاوز نشد."}

    # Estimate where weights will reside. Prefer Ollama's observed size_vram when loaded.
    if gpu:
        gpu_budget = max(0, int(gpu["total"] * VRAM_USABLE) - int(other_vram_bytes) - OVERHEAD_BYTES)
        if model_vram_bytes is not None:
            gpu_weights = max(0, min(int(model_vram_bytes), int(model_bytes)))
        else:
            gpu_weights = min(int(model_bytes), int(gpu_budget * 0.85))
        gpu_kv_budget = max(0, gpu_budget - gpu_weights)
    else:
        gpu_weights = 0
        gpu_kv_budget = 0

    if ram:
        # Keep OS, Python, browser and transient model buffers out of the KV-cache budget.
        reserve = int(reserve_ram_bytes if reserve_ram_bytes is not None else 8 * _GB)
        cpu_weights = max(0, int(model_bytes) - gpu_weights)
        ram_kv_budget = max(0, int(ram.get("available", 0)) - reserve - cpu_weights - OVERHEAD_BYTES)
    else:
        cpu_weights = 0
        ram_kv_budget = 0

    # Approximate KV-cache placement by the fraction of model weights offloaded to each device.
    # If a pool is absent, its share is treated as zero rather than silently overcommitting it.
    gpu_fraction = (gpu_weights / model_bytes) if gpu and model_bytes else 0.0
    cpu_fraction = (cpu_weights / model_bytes) if ram and model_bytes else 0.0
    if gpu_fraction + cpu_fraction <= 0:
        return None
    max_by_gpu = (gpu_kv_budget / (kv * gpu_fraction * KV_SAFETY)) if gpu_fraction > 0 else float("inf")
    max_by_ram = (ram_kv_budget / (kv * cpu_fraction * KV_SAFETY)) if cpu_fraction > 0 else float("inf")
    max_tokens = min(limit, int(min(max_by_gpu, max_by_ram)))
    chosen = max((step for step in STEPS if step <= max_tokens), default=None)
    if chosen is None:
        chosen = min(min_context, limit)
        reason = "منابع حافظه محدود است؛ حداقل پنجرهٔ امن انتخاب شد و بخشی از مدل روی CPU قرار می‌گیرد."
        fits_gpu = False
    else:
        fits_gpu = bool(gpu and gpu_fraction >= 0.999 and chosen * kv * KV_SAFETY <= gpu_kv_budget)
        reason = ("بر اساس RAM و VRAM آزاد و نسبت بارگذاری مدل محاسبه شد." if not fits_gpu
                  else "بزرگ‌ترین پنجرهٔ برآوردشده که کنار وزن‌های مدل در VRAM جا می‌شود.")
    gpu_kv = chosen * kv * gpu_fraction
    cpu_kv = chosen * kv * cpu_fraction
    return {"num_ctx": chosen, "fits_gpu": fits_gpu,
            "kv_mb": round((gpu_kv + cpu_kv) / _MIB),
            "gpu_kv_mb": round(gpu_kv / _MIB), "ram_kv_mb": round(cpu_kv / _MIB),
            "gpu_weight_fraction": round(gpu_fraction, 3), "reason": reason}


def cpu_info():
    """CPU model name and core counts (None fields when the platform does not say)."""
    name = platform.processor() or None
    if name and os.name != "nt" and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                name = next((l.split(":", 1)[1].strip() for l in handle if l.lower().startswith("model name")), name)
        except OSError:
            pass
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        pass
    return {"name": name, "logical": os.cpu_count(), "physical": physical}


def snapshot(ttl=20):
    """One consistent view of the machine for planning and diagnostics.

    ``ram`` / ``gpu`` are None when unmeasurable; ``unavailable`` names what is missing so the UI can say so instead of
    showing a made-up zero. Dynamic values (free memory) are only as fresh as ``ttl`` seconds.
    """
    ram, gpu = ram_info(), gpu_info(ttl)
    gpus = [g for g in all_gpus(ttl)] if gpu is not None or ram is not None else []
    unavailable = []
    if ram is None:
        unavailable.append("ram")
    if gpu is None:
        unavailable.append("gpu")
    elif gpu.get("used") is None:
        unavailable.append("gpu_usage")
    return {"ram": ram, "gpu": gpu, "gpus": gpus, "cpu": cpu_info(), "unavailable": unavailable,
            "taken_at": time.time()}


def describe(gpu=None, ram=None):
    """UI-friendly summary (GPU usage fields are None when the platform cannot report them)."""
    gpu = gpu if gpu is not None else gpu_info()
    ram = ram if ram is not None else ram_info()
    gb = lambda value: round(value / _GB, 1) if isinstance(value, (int, float)) else None
    return {
        "gpu": ({"name": gpu["name"], "total_gb": gb(gpu["total"]), "free_gb": gb(gpu.get("free")),
                 "used_gb": gb(gpu.get("used")), "source": gpu.get("source", "nvidia-smi")} if gpu else None),
        "ram": ({"total_gb": gb(ram["total"]), "available_gb": gb(ram["available"]),
                 "used_gb": gb(ram["total"] - ram["available"]), "source": ram.get("source", "psutil")} if ram else None),
    }
