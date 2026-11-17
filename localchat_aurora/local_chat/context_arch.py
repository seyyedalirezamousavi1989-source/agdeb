"""Architecture adapters: how much memory a context window costs, per model *architecture*.

A single «layers × kv_heads × head_dim × 2 × bytes» formula is only right for a conventional full-attention
transformer. Hybrid models (full attention every N layers, the rest linear-attention / state-space) keep a KV cache
in a fraction of their layers and a small *fixed-size* recurrent state in the others. Applying the textbook formula
to every layer over-counts such a model several times over, which is what used to shrink its window.

Adapters are small, independent and selected from the *metadata* that ``/api/show`` reports (never from the model
name), so a new architecture is one new class. Each adapter answers two questions:

* ``applies(meta)``   – does the metadata describe this kind of model AND is it complete enough to compute from?
* ``estimate(meta, ctx, cache_elem_bytes)`` – bytes for the context-dependent KV cache and for the fixed state.

Every number here is an *estimate derived from metadata*, not a measurement; ``CacheEstimate.confidence`` says how
far it can be trusted:

``estimated``    computed from complete metadata with a known formula;
``upper_bound``  computed from metadata but known to over-count (e.g. sliding-window / compressed-KV models);
``unsupported``  metadata is missing or unrecognised, no formula applies – the caller must fall back to observation.
"""
from dataclasses import dataclass, field

# Bytes per cache element for Ollama's selectable KV-cache types (OLLAMA_KV_CACHE_TYPE).
KV_ELEMENT_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
DEFAULT_KV_TYPE = "f16"           # Ollama's default; also the most pessimistic choice
RECURRENT_STATE_BYTES = 4         # recurrent / convolution state is kept in float32 by the runtimes


@dataclass(frozen=True)
class CacheEstimate:
    kv_bytes: int                 # grows linearly with the context
    state_bytes: int              # fixed recurrent/convolution state, independent of the context
    confidence: str               # "estimated" | "upper_bound" | "unsupported"
    method: str                   # adapter name
    notes: tuple = ()
    details: dict = field(default_factory=dict)

    @property
    def total_bytes(self):
        return self.kv_bytes + self.state_bytes

    @property
    def supported(self):
        return self.confidence != "unsupported"


def _positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _scalar(value):
    """A plain positive number; per-layer lists are handled by the callers that understand them."""
    return int(value) if _positive_number(value) else None


def _per_layer(value, layers):
    """A per-layer list of non-negative numbers of exactly ``layers`` entries (e.g. KV heads per layer), else None."""
    if (isinstance(value, list) and layers and len(value) == layers
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in value)):
        return [int(v) for v in value]
    return None


def _is_list(value):
    return isinstance(value, list)


class ArchitectureAdapter:
    name = "base"

    def applies(self, meta):  # pragma: no cover - interface
        raise NotImplementedError

    def estimate(self, meta, ctx, cache_elem_bytes):  # pragma: no cover - interface
        raise NotImplementedError


def _attention_geometry(meta):
    """(kv_heads, key_len, value_len, notes) from attention metadata, or None when it cannot be derived.

    ``kv_heads`` may be an int or a per-layer list (hybrid models publish 0 for layers without attention).
    """
    notes = []
    heads = meta.get("attention.head_count")
    kv = meta.get("attention.head_count_kv")
    if kv is None or (not _positive_number(kv) and not _is_list(kv)):
        if _positive_number(heads):
            kv = heads
            notes.append("head_count_kv missing: multi-head attention assumed (upper bound)")
        else:
            return None
    key_len = _scalar(meta.get("attention.key_length"))
    value_len = _scalar(meta.get("attention.value_length")) or key_len
    if not key_len:
        width, head_count = _scalar(meta.get("embedding_length")), _scalar(heads)
        if not (width and head_count):
            return None
        key_len = value_len = width // head_count
        notes.append("key_length missing: derived as embedding_length / head_count")
    return kv, key_len, value_len or key_len, notes


class HybridRecurrentAdapter(ArchitectureAdapter):
    """Full attention every ``full_attention_interval`` layers, linear-attention / state-space in the others.

    KV cache exists only in the full-attention layers; every other layer keeps a constant-size recurrent state.
    Selected by the *presence of the hybrid metadata*, so it covers any family that publishes it (qwen35 and
    similar), not one model name.
    """
    name = "hybrid_recurrent"

    def applies(self, meta):
        layers = _scalar(meta.get("block_count"))
        has_state = all(_scalar(meta.get(k)) for k in ("ssm.inner_size", "ssm.state_size", "ssm.conv_kernel"))
        if not (layers and has_state and _attention_geometry(meta)):
            return False
        # Either an interval says which layers are full attention, or an explicit per-layer KV-head list does.
        return bool(_scalar(meta.get("full_attention_interval"))
                    or _per_layer(meta.get("attention.head_count_kv"), layers))

    def layer_split(self, meta):
        """(full_attention_layers, recurrent_layers, notes) following the ``(i + 1) % interval == 0`` convention."""
        layers = _scalar(meta.get("block_count"))
        interval = _scalar(meta.get("full_attention_interval"))
        if not interval:
            listed = _per_layer(meta.get("attention.head_count_kv"), layers)
            full = sum(1 for h in listed if h > 0)
            return full, layers - full, ["attention layers taken from the per-layer head_count_kv list"]
        nextn = _scalar(meta.get("nextn_predict_layers")) or 0
        main = max(0, layers - nextn)
        full = sum(1 for i in range(main) if (i + 1) % interval == 0)
        notes = []
        if nextn:
            # The multi-token-prediction layer is an attention layer; count it so the estimate errs high.
            full += nextn
            notes.append(f"{nextn} next-token-prediction layer(s) counted as full attention (conservative)")
        return full, main - (full - nextn), notes

    def estimate(self, meta, ctx, cache_elem_bytes):
        kv_heads, key_len, value_len, notes = _attention_geometry(meta)
        notes = list(notes)
        full, recurrent, split_notes = self.layer_split(meta)
        notes += split_notes
        layers = _scalar(meta.get("block_count"))
        per_layer = _per_layer(kv_heads, layers)
        if per_layer is not None:
            # An explicit per-layer list is authoritative: count only layers that really have KV heads.
            kv_per_token = sum(h * (key_len + value_len) for h in per_layer) * cache_elem_bytes
        else:
            heads = int(max(kv_heads)) if _is_list(kv_heads) else int(kv_heads)
            kv_per_token = full * heads * (key_len + value_len) * cache_elem_bytes
        inner, state_size, kernel = (int(meta["ssm.inner_size"]), int(meta["ssm.state_size"]),
                                     int(meta["ssm.conv_kernel"]))
        groups = _scalar(meta.get("ssm.group_count")) or 0
        conv_dim = inner + 2 * groups * state_size
        per_recurrent_layer = ((kernel - 1) * conv_dim + state_size * inner) * RECURRENT_STATE_BYTES
        notes.append("recurrent state size derived from ssm.* metadata (gated delta-net layout assumed)")
        return CacheEstimate(
            kv_bytes=int(kv_per_token * ctx), state_bytes=int(per_recurrent_layer * recurrent),
            confidence="estimated", method=self.name, notes=tuple(notes),
            details={"full_attention_layers": full, "recurrent_layers": recurrent,
                     "kv_bytes_per_token": int(kv_per_token), "state_bytes_per_layer": int(per_recurrent_layer)})


class FullAttentionAdapter(ArchitectureAdapter):
    """Conventional transformer: multi-head, multi-query or grouped-query attention in every layer."""
    name = "full_attention"

    def applies(self, meta):
        return bool(_scalar(meta.get("block_count")) and _attention_geometry(meta)
                    and "ssm.state_size" not in meta and "full_attention_interval" not in meta)

    def estimate(self, meta, ctx, cache_elem_bytes):
        layers = _scalar(meta.get("block_count"))
        kv_heads, key_len, value_len, notes = _attention_geometry(meta)
        notes = list(notes)
        confidence = "estimated"
        per_layer = _per_layer(kv_heads, layers)
        if per_layer is not None:
            kv_per_token = sum(h * (key_len + value_len) for h in per_layer) * cache_elem_bytes
        else:
            heads = int(max(kv_heads)) if _is_list(kv_heads) else int(kv_heads)
            if _is_list(kv_heads):
                notes.append("per-layer head_count_kv list does not match block_count: maximum used (upper bound)")
                confidence = "upper_bound"
            kv_per_token = layers * heads * (key_len + value_len) * cache_elem_bytes
        # Models whose KV cache is smaller than the textbook formula: the formula is then an upper bound.
        if _scalar(meta.get("attention.sliding_window")):
            notes.append("sliding-window attention: full-context cache assumed in every layer (upper bound)")
            confidence = "upper_bound"
        if _scalar(meta.get("attention.kv_lora_rank")):
            notes.append("compressed (latent) KV attention: expanded cache assumed (upper bound)")
            confidence = "upper_bound"
        return CacheEstimate(kv_bytes=int(kv_per_token * ctx), state_bytes=0, confidence=confidence,
                             method=self.name, notes=tuple(notes),
                             details={"full_attention_layers": layers, "recurrent_layers": 0,
                                      "kv_bytes_per_token": int(kv_per_token)})


class UnsupportedAdapter(ArchitectureAdapter):
    """Anything else (missing/unknown metadata, pure SSM models): never guess a formula."""
    name = "unsupported"

    def applies(self, meta):
        return True

    def estimate(self, meta, ctx, cache_elem_bytes):
        return CacheEstimate(0, 0, "unsupported", self.name,
                             ("metadata is missing or describes an unrecognised architecture",), {})


# Order matters: the most specific adapter first, the catch-all last.
ADAPTERS = (HybridRecurrentAdapter(), FullAttentionAdapter(), UnsupportedAdapter())


def select_adapter(meta):
    meta = meta if isinstance(meta, dict) else {}
    return next(adapter for adapter in ADAPTERS if adapter.applies(meta))


def estimate_cache(meta, ctx, kv_type=DEFAULT_KV_TYPE):
    """Cache cost of ``ctx`` tokens for a model described by ``meta`` (see :class:`CacheEstimate`)."""
    elem = KV_ELEMENT_BYTES.get(kv_type, KV_ELEMENT_BYTES[DEFAULT_KV_TYPE])
    return select_adapter(meta).estimate(meta if isinstance(meta, dict) else {}, max(1, int(ctx)), elem)
