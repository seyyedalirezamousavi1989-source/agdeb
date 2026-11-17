"""Adaptive context: metadata, runtime, hardware, estimator/allocator, reload policy, monitor and recovery."""
import ctypes
import json
import unittest
from unittest.mock import patch

from local_chat import adaptive_context as ac
from local_chat import context_arch, context_manager as cm, hardware, model_inspector as mi, runtime_inspector as ri

GIB = 1024 ** 3
QWEN35 = {"block_count": 65, "context_length": 262144, "embedding_length": 5120, "attention.head_count": 24,
          "attention.head_count_kv": 4, "attention.key_length": 256, "attention.value_length": 256,
          "full_attention_interval": 4, "nextn_predict_layers": 1, "ssm.conv_kernel": 4, "ssm.group_count": 16,
          "ssm.inner_size": 6144, "ssm.state_size": 128, "ssm.time_step_rank": 48}


def show_json(arch="qwen35", meta=None, **extra):
    info = {"general.architecture": arch, "general.parameter_count": 27300000000}
    info.update({f"{arch}.{k}": v for k, v in (QWEN35 if meta is None else meta).items()})
    return {"model_info": info, "details": {"family": arch, "parameter_size": "27.3B", "quantization_level": "Q4_K_M"},
            "parameters": "num_ctx 8192\nstop x", "capabilities": ["completion", "tools"], **extra}


def profile(meta=None, size=18410394087, name="m:27b", arch="qwen35"):
    return mi.parse_show(name, show_json(arch, meta), {"size": size, "digest": "d"})


def gpu(total=12 * GIB, used=1 * GIB):
    return {"name": "NVIDIA RTX 3060", "total": total, "used": used, "free": total - used, "dedicated": True, "source": "nvml"}


def ram(total=int(47.75 * GIB), available=int(44 * GIB)):
    return {"total": total, "available": available, "used": total - available, "source": "psutil"}


class ModelMetadataTests(unittest.TestCase):
    def test_complete_hybrid_metadata(self):
        p = profile()
        self.assertEqual((p.architecture, p.declared_context, p.quantization, p.file_size), ("qwen35", 262144, "Q4_K_M", 18410394087))
        self.assertEqual(p.parameter_count, 27300000000)
        self.assertTrue(p.hybrid)
        self.assertTrue(p.complete, p.issues)
        self.assertEqual(p.modelfile_num_ctx, 8192)

    def test_missing_metadata_is_reported_not_guessed(self):
        p = mi.parse_show("x", {}, {})
        self.assertIsNone(p.declared_context)
        self.assertIn("/api/show returned no usable data", p.issues)
        self.assertFalse(p.complete)

    def test_invalid_metadata_types(self):
        show = show_json(meta={"block_count": "sixty", "context_length": True, "attention.head_count": [1, "x"]})
        p = mi.parse_show("x", show, {})
        self.assertIsNone(p.declared_context)
        self.assertNotIn("block_count", p.arch_meta)
        self.assertTrue(any("unsupported type" in i or "not a numeric list" in i for i in p.issues))
        self.assertEqual(mi.parse_show("x", {"model_info": "oops"}, {}).architecture, None)

    def test_unknown_architecture_has_no_cache_formula(self):
        p = profile(meta={"context_length": 4096}, arch="weirdnet")
        self.assertEqual(p.declared_context, 4096)
        self.assertFalse(context_arch.estimate_cache(p.arch_meta, 4096).supported)

    def test_other_families_use_their_own_prefix(self):
        llama = {"block_count": 32, "context_length": 131072, "attention.head_count": 32, "attention.head_count_kv": 8,
                 "embedding_length": 4096}
        p = profile(meta=llama, arch="llama")
        est = context_arch.estimate_cache(p.arch_meta, 1000)
        self.assertEqual(est.method, "full_attention")
        self.assertEqual(est.kv_bytes, 32 * 8 * (128 + 128) * 2 * 1000)   # key_length derived 4096/32
        self.assertFalse(p.hybrid)

    def test_parameter_size_parsing(self):
        self.assertEqual(mi.parse_parameter_size("27.3B"), 27300000000)
        self.assertIsNone(mi.parse_parameter_size("lots"))

    def test_inspector_caches_and_survives_failures(self):
        calls = []
        def show(name):
            calls.append(name)
            if len(calls) == 1:
                raise OSError("down")
            return show_json()
        clock = [0.0]
        inspector = mi.ModelInspector(show, lambda: [{"name": "m", "size": 5}], clock=lambda: clock[0])
        first = inspector.inspect("m")
        self.assertFalse(first.complete)
        self.assertIn("/api/show failed", first.issues[0])
        inspector.inspect("m")
        self.assertEqual(len(calls), 1)            # the failure is cached briefly, not retried per request
        clock[0] += mi.FAILURE_TTL + 1
        self.assertTrue(inspector.inspect("m").hybrid)
        inspector.inspect("m")
        self.assertEqual(len(calls), 2)            # success cached


class ArchitectureTests(unittest.TestCase):
    def test_hybrid_is_not_costed_as_full_attention_in_every_layer(self):
        est = context_arch.estimate_cache(QWEN35, 262144)
        naive = 65 * 4 * 512 * 2 * 262144
        self.assertEqual(est.method, "hybrid_recurrent")
        self.assertEqual(est.details["full_attention_layers"], 17)      # 16 by interval + 1 next-token layer
        self.assertEqual(est.details["recurrent_layers"], 48)
        self.assertLess(est.kv_bytes, naive / 3)
        self.assertEqual(est.confidence, "estimated")

    def test_recurrent_state_is_fixed_size(self):
        small, big = context_arch.estimate_cache(QWEN35, 1024), context_arch.estimate_cache(QWEN35, 262144)
        self.assertEqual(small.state_bytes, big.state_bytes)
        self.assertGreater(big.state_bytes, 0)
        self.assertEqual(big.kv_bytes, 256 * small.kv_bytes)

    def test_cache_type_changes_kv_only(self):
        f16 = context_arch.estimate_cache(QWEN35, 8192, "f16")
        q8 = context_arch.estimate_cache(QWEN35, 8192, "q8_0")
        self.assertLess(q8.kv_bytes, f16.kv_bytes)
        self.assertEqual(q8.state_bytes, f16.state_bytes)

    def test_per_layer_kv_list_counts_only_attention_layers(self):
        meta = {"block_count": 4, "attention.head_count": 8, "attention.head_count_kv": [0, 2, 0, 2],
                "attention.key_length": 64, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.conv_kernel": 4}
        est = context_arch.estimate_cache(meta, 10)
        self.assertEqual(est.method, "hybrid_recurrent")
        self.assertEqual(est.kv_bytes, 2 * 2 * (64 + 64) * 2 * 10)
        self.assertEqual(est.details["recurrent_layers"], 2)

    def test_sliding_window_and_missing_kv_heads_are_upper_bounds(self):
        meta = {"block_count": 10, "attention.head_count": 8, "attention.key_length": 64, "attention.sliding_window": 1024}
        est = context_arch.estimate_cache(meta, 100)
        self.assertEqual(est.confidence, "upper_bound")
        self.assertTrue(any("multi-head" in n for n in est.notes))

    def test_incomplete_or_pure_ssm_is_unsupported(self):
        self.assertFalse(context_arch.estimate_cache({}, 10).supported)
        self.assertFalse(context_arch.estimate_cache({"block_count": 4, "ssm.state_size": 16}, 10).supported)
        self.assertFalse(context_arch.estimate_cache(None, 10).supported)


class RuntimeTests(unittest.TestCase):
    PS = {"models": [{"name": "m:27b", "size": 19 * GIB, "size_vram": 9567815596, "context_length": 8192},
                     {"bogus": 1}, "junk"]}

    def test_parse_ps_skips_malformed_entries(self):
        models = ri.parse_ps(self.PS)
        self.assertEqual([m.name for m in models], ["m:27b"])
        self.assertAlmostEqual(models[0].gpu_fraction, 9567815596 / (19 * GIB))
        self.assertEqual(ri.parse_ps(None), [])

    def test_declared_requested_actual_are_kept_apart(self):
        loaded = ri.parse_ps(self.PS)[0]
        t = ri.compare(262144, 32768, loaded)
        self.assertEqual((t.declared, t.requested, t.actual, t.status), (262144, 32768, 8192, "mismatch"))
        self.assertTrue(t.reload_required)
        self.assertFalse(t.public()["verified"])
        self.assertEqual(ri.compare(262144, 8192, loaded).status, "verified")

    def test_unloaded_unknown_and_unreachable(self):
        self.assertEqual(ri.compare(1, 2, None).status, "not_loaded")
        self.assertEqual(ri.compare(1, 2, ri.LoadedModel("m")).status, "unknown")   # no context_length reported
        self.assertEqual(ri.compare(1, 2, None, reachable=False).status, "unreachable")

    def test_unreachable_ollama_is_not_an_empty_list(self):
        def down():
            raise ConnectionRefusedError("no")
        state = ri.RuntimeInspector(down).read()
        self.assertFalse(state.reachable)
        self.assertEqual(state.models, ())
        self.assertTrue(ri.RuntimeInspector(lambda: {"models": []}).read().reachable)

    def test_ps_is_cached_briefly(self):
        calls = []
        clock = [0.0]
        inspector = ri.RuntimeInspector(lambda: calls.append(1) or self.PS, clock=lambda: clock[0])
        inspector.read(); inspector.read()
        self.assertEqual(len(calls), 1)
        inspector.read(fresh=True)
        self.assertEqual(len(calls), 2)

    def test_model_name_match_ignores_latest_tag(self):
        self.assertTrue(ri.same_model("Qwen:LATEST".replace("LATEST", "latest"), "qwen"))


class HardwareTests(unittest.TestCase):
    def test_nvml_gpu(self):
        class Lib:
            def nvmlInit_v2(self): return 0
            def nvmlShutdown(self): return 0
            def nvmlDeviceGetCount_v2(self, ref): ref._obj.value = 1; return 0
            def nvmlDeviceGetHandleByIndex_v2(self, i, ref): return 0
            def nvmlDeviceGetMemoryInfo(self, handle, ref):
                ref._obj.total, ref._obj.free, ref._obj.used = 12 * GIB, 10 * GIB, 2 * GIB
                return 0
            def nvmlDeviceGetName(self, handle, buf, n): buf.value = b"NVIDIA GeForce RTX 3060"; return 0
        gpus = hardware._query_nvml(Lib())
        self.assertEqual((gpus[0]["name"], gpus[0]["total"], gpus[0]["used"], gpus[0]["free"], gpus[0]["source"]),
                         ("NVIDIA GeForce RTX 3060", 12 * GIB, 2 * GIB, 10 * GIB, "nvml"))
        self.assertEqual(gpus[0]["vendor"], "nvidia")

    def test_nvml_failure_returns_nothing(self):
        class Broken:
            def nvmlInit_v2(self): return 1
        self.assertEqual(hardware._query_nvml(Broken()), [])

    def test_windows_adapters_dedicated_and_integrated(self):
        text = json.dumps({"adapters": [{"name": "Intel(R) UHD Graphics 730", "total": 128 * 1024 ** 2},
                                        {"name": "NVIDIA GeForce RTX 3060", "total": 12 * GIB}],
                           "dedicated_usage": 3 * GIB})
        gpus = hardware._parse_windows_gpus(text)
        by = {g["name"]: g for g in gpus}
        self.assertFalse(by["Intel(R) UHD Graphics 730"]["dedicated"])
        self.assertEqual(by["NVIDIA GeForce RTX 3060"]["used"], 3 * GIB)
        self.assertEqual(hardware._parse_windows_gpus("not json"), [])

    def test_windows_usage_not_attributed_when_ambiguous(self):
        text = json.dumps({"adapters": [{"name": "AMD Radeon RX 6800", "total": 16 * GIB},
                                        {"name": "NVIDIA RTX 4090", "total": 24 * GIB}], "dedicated_usage": 5 * GIB})
        self.assertTrue(all(g["used"] is None and g["free"] is None for g in hardware._parse_windows_gpus(text)))

    def test_no_dedicated_gpu_and_missing_telemetry(self):
        with patch.object(hardware, "all_gpus", return_value=[]), patch.object(hardware, "ram_info", return_value=None):
            snap = hardware.snapshot()
        self.assertIsNone(snap["gpu"])
        self.assertEqual(snap["unavailable"], ["ram", "gpu"])
        with patch.object(hardware, "all_gpus", return_value=[hardware._gpu_entry("Intel UHD Graphics", 1, 0, "x")]):
            self.assertIsNone(hardware.gpu_info(ttl=0))     # an integrated GPU is never treated as VRAM

    def test_unmeasured_vram_usage_is_reported_unavailable(self):
        entry = hardware._gpu_entry("NVIDIA X", 8 * GIB, None, "windows-registry", dedicated=True)
        with patch.object(hardware, "all_gpus", return_value=[entry]), patch.object(hardware, "ram_info", return_value=ram()):
            snap = hardware.snapshot()
        self.assertIn("gpu_usage", snap["unavailable"])
        self.assertIsNone(hardware.describe(entry, ram())["gpu"]["free_gb"])

    def test_ram_fallback_without_psutil(self):
        import builtins
        real = builtins.__import__
        def no_psutil(name, *a, **k):
            if name == "psutil":
                raise ImportError
            return real(name, *a, **k)
        with patch("builtins.__import__", no_psutil):
            info = hardware.ram_info()
        self.assertTrue(info is None or info["source"] != "psutil")


class AllocatorTests(unittest.TestCase):
    def plan(self, p=None, g=None, r=None, runtime=None, **kw):
        hw = {"ram": r or ram(), "gpu": g if g is not None else gpu()}
        return ac.plan_context(p or profile(), hw, runtime or ri.RuntimeState(True, ()), **kw)

    def test_target_is_not_the_declared_maximum(self):
        plan = self.plan()
        self.assertLess(plan.target, 262144)
        self.assertEqual(plan.limiting, "performance")           # fits in memory far beyond the chosen window
        self.assertGreater(plan.max_feasible, plan.target)
        self.assertEqual(plan.confidence, "estimated")

    def test_user_ceiling_is_respected(self):
        plan = self.plan(ceiling=16384)
        self.assertEqual((plan.target, plan.limiting), (16384, "ceiling"))

    def test_small_model_reaches_its_declared_limit(self):
        small = profile(meta={**QWEN35, "context_length": 40960, "full_attention_interval": None}, size=4 * GIB)
        meta = {"block_count": 36, "context_length": 40960, "attention.head_count": 32, "attention.head_count_kv": 8,
                "attention.key_length": 128, "embedding_length": 4096}
        small = profile(meta=meta, size=4 * GIB, arch="qwen3")
        plan = self.plan(small, g=gpu(24 * GIB))
        self.assertEqual((plan.target, plan.limiting), (40960, "declared"))
        self.assertTrue(plan.fits_gpu)

    def test_plan_is_identical_whether_model_is_loaded_or_not(self):
        cold = self.plan()
        loaded = ri.LoadedModel("m:27b", size=19 * GIB, size_vram=9567815596, context_length=8192)
        warm = self.plan(g=gpu(used=int(10.4 * GIB)), r=ram(available=int(35.11 * GIB)),
                         runtime=ri.RuntimeState(True, (loaded,)))
        self.assertEqual(cold.target, warm.target)

    def test_insufficient_vram_and_low_ram(self):
        tiny = self.plan(g=gpu(4 * GIB), r=ram(total=16 * GIB, available=int(11 * GIB)))
        self.assertEqual(tiny.limiting, "memory_insufficient")
        self.assertEqual(tiny.target, 4096)
        lowram = self.plan(g=gpu(4 * GIB), r=ram(total=24 * GIB, available=int(14 * GIB)))
        self.assertLessEqual(lowram.target, 16384)

    def test_other_gpu_consumers_shrink_the_budget(self):
        free = self.plan(g=gpu(used=1 * GIB), ceiling=262144)
        busy = self.plan(g=gpu(used=int(8 * GIB)), ceiling=262144)
        self.assertLess(busy.gpu_capacity, free.gpu_capacity)
        self.assertGreater(busy.cpu_bytes, free.cpu_bytes)        # less VRAM left → more of the model on the CPU

    def test_no_gpu_data_and_not_loaded_returns_none(self):
        self.assertIsNone(ac.plan_context(profile(), {"ram": ram(), "gpu": None}, ri.RuntimeState(True, ())))
        self.assertIsNone(ac.plan_context(profile(size=None), {"ram": ram(), "gpu": gpu()}, ri.RuntimeState(True, ())))

    def test_no_gpu_but_loaded_model_plans_from_runtime(self):
        loaded = ri.LoadedModel("m:27b", size=20 * GIB, size_vram=0, context_length=8192)
        plan = ac.plan_context(profile(), {"ram": ram(), "gpu": None}, ri.RuntimeState(True, (loaded,)), ceiling=32768)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.gpu_capacity, 0)

    def test_unsupported_architecture_uses_labelled_fallback(self):
        unknown = profile(meta={"context_length": 32768}, arch="weirdnet")
        plan = self.plan(unknown)
        self.assertTrue(plan.confidence.startswith("fallback"))
        self.assertTrue(any("fallback" in n for n in plan.notes))

    def test_observed_runtime_drives_fallback_estimate(self):
        unknown = profile(meta={"context_length": 32768}, arch="weirdnet")
        loaded = ri.LoadedModel("m:27b", size=int(18.41 * GIB) + 2 * GIB, size_vram=0, context_length=8192)
        cost = ac.estimate_cost(unknown, 16384, observed=(loaded.context_length, loaded.size))
        self.assertEqual(cost.confidence, "fallback_observed")

    def test_calibration_only_raises_the_estimate(self):
        cal, p = ac.Calibration(), profile()
        base = ac.estimate_cost(p, 8192)
        cal.observe("m", p, 8192, p.file_size + base.extra_bytes * 2)       # Ollama allocated twice the prediction
        self.assertAlmostEqual(cal.factor("m"), 2.0, places=1)
        cal.observe("m", p, 8192, p.file_size + base.extra_bytes // 2)      # smaller observation never lowers it
        self.assertAlmostEqual(cal.factor("m"), 2.0, places=1)
        self.assertGreater(ac.estimate_cost(p, 8192, calibration=cal.factor("m")).extra_bytes, base.extra_bytes)

    def test_window_steps_include_odd_declared_limit(self):
        steps = ac.window_steps(40960)
        self.assertEqual(steps[-1], 40960)
        self.assertIn(32768, steps)

    def test_env_cache_settings(self):
        self.assertEqual(ac.env_cache_settings({})["kv_type"], "f16")
        self.assertEqual(ac.env_cache_settings({"OLLAMA_KV_CACHE_TYPE": "q8_0"})["kv_type"], "f16")   # needs flash attention
        env = {"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "x"}
        self.assertEqual(ac.env_cache_settings(env), {"kv_type": "q8_0", "parallel": 1})


class ReloadPolicyTests(unittest.TestCase):
    def setUp(self):
        self.plan = ac.plan_context(profile(), {"ram": ram(), "gpu": gpu()}, ri.RuntimeState(True, ()), ceiling=32768)
        self.coordinator = ac.ContextCoordinator(clock=lambda: 0.0)

    def test_cold_load_uses_target_and_is_not_a_reload(self):
        d = self.coordinator.decide("m", self.plan, None)
        self.assertEqual((d.action, d.num_ctx, d.reload), ("cold_load", 32768, False))

    def test_concurrent_request_follows_pending_load(self):
        self.coordinator.decide("m", self.plan, None)
        other = ac.ContextPlan(**{**self.plan.__dict__, "target": 4096})
        d = self.coordinator.decide("m", other, None)
        self.assertEqual((d.action, d.num_ctx), ("follow_pending", 32768))

    def test_small_request_keeps_resident_window(self):
        d = self.coordinator.decide("m", self.plan, 8192, required_tokens=1000)
        self.assertEqual((d.action, d.num_ctx, d.reload), ("keep", 8192, False))

    def test_grows_once_when_window_is_getting_full(self):
        d = self.coordinator.decide("m", self.plan, 8192, required_tokens=int(8192 * ac.GROW_AT) + 10)
        self.assertEqual((d.action, d.num_ctx, d.reload), ("grow", 32768, True))

    def test_shrinks_when_resident_window_exceeds_ceiling_or_memory(self):
        d = self.coordinator.decide("m", self.plan, 65536)
        self.assertEqual((d.action, d.num_ctx, d.reload), ("shrink", 32768, True))

    def test_pending_expires(self):
        clock = [0.0]
        c = ac.ContextCoordinator(clock=lambda: clock[0])
        c.decide("m", self.plan, None)
        clock[0] = ac.PENDING_TTL + 1
        self.assertEqual(c.decide("m", self.plan, None).action, "cold_load")


class MonitorAndRecoveryTests(unittest.TestCase):
    def test_estimates_are_flagged_and_measured_usage_is_exact(self):
        ok, budget = cm.guard_request([{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}], 8192, 1024)
        info = budget.as_dict()
        self.assertFalse(info["exact"])
        self.assertEqual(info["remaining"], 8192 - budget.used)
        m = cm.measured_usage({"prompt_eval_count": 500, "eval_count": 100, "done": True}, 8192)
        self.assertEqual((m["used"], m["remaining"], m["exact"]), (600, 7592, True))
        self.assertTrue(m["prompt_may_exclude_cache"])
        self.assertIsNone(cm.measured_usage({"done": True}, 8192))
        self.assertIsNone(cm.measured_usage(None, 8192))

    def test_calibrator_learns_undercount_but_never_lowers_estimate(self):
        cal = cm.TokenCalibrator()
        cal.observe("m", 1000, 1400)
        self.assertAlmostEqual(cal.ratio("m"), 1.4)
        cal.observe("m", 1000, 300)                 # cached prompt → low count must not lower the estimate below 1
        self.assertGreaterEqual(cal.ratio("m"), 1.0)
        cal.observe("m", 0, 5); cal.observe("m", 5, None); cal.observe("m", 5, True)
        self.assertGreater(cal.adjust("m", 1000), 1000)

    def test_token_ratio_scales_guard(self):
        msgs = [{"role": "user", "content": "x" * 4000}]
        _, plain = cm.guard_request(msgs, 8192, 512)
        _, scaled = cm.guard_request(msgs, 8192, 512, token_ratio=1.5)
        self.assertGreater(scaled.used, plain.used)

    def test_tool_payload_counted_once(self):
        tools = [{"type": "function", "function": {"name": "t", "description": "d" * 400, "parameters": {}}}]
        _, with_tools = cm.guard_request([{"role": "user", "content": "hi"}], 8192, 512, tools)
        _, without = cm.guard_request([{"role": "user", "content": "hi"}], 8192, 512)
        self.assertEqual(with_tools.history_tokens, without.history_tokens)
        self.assertGreater(with_tools.tools_tokens, 100)

    def test_char_budget_grows_for_english_not_for_persian(self):
        p = {"num_ctx": 32768, "num_predict": 2048, "budget": int(30720 * 2.1)}
        english = cm.adapt_char_budget(p, "the quick brown fox jumps over the lazy dog " * 40)
        persian = cm.adapt_char_budget(p, "سلام این یک جمله فارسی طولانی است " * 40)
        self.assertGreater(english["budget"], p["budget"] * 1.4)
        self.assertEqual(persian["budget"], p["budget"])
        self.assertEqual(cm.adapt_char_budget(p, "")["budget"], p["budget"])

    def test_context_overflow_prevention_and_recovery_report(self):
        big = [{"role": "user", "content": "word " * 9000}, {"role": "assistant", "content": "word " * 9000}]
        ok, before = cm.guard_request(big, 8192, 1024)
        self.assertFalse(ok)                                   # would overflow: must be caught before sending
        small = [big[0]["content"][:4000]]
        ok2, after = cm.guard_request([{"role": "user", "content": small[0]}], 8192, 1024)
        report = cm.recovery_report(before.used, after, 2)
        self.assertTrue(report["ok"])
        self.assertGreater(report["recovered"], 0)
        self.assertEqual(report["before"], before.used)
        failed = cm.recovery_report(before.used, before, 3)
        self.assertFalse(failed["ok"])

    def test_long_conversation_compresses_without_losing_decisions(self):
        history = []
        for i in range(60):
            history += [{"role": "user", "content": f"سوال {i} " + "متن " * 120},
                        {"role": "assistant", "content": f"پاسخ {i} " + "متن " * 120}]
        history[0]["content"] = "مهم: همیشه فارسی جواب بده و از پایگاه‌داده SQLite استفاده کن. " + "x" * 50
        older = history[:-6]
        digest = cm.history_digest(older, 1500)
        pinned = cm.pinned_digest(older)
        self.assertTrue(digest)
        self.assertIn("SQLite", pinned)            # the standing instruction survives compression of the old turns
        self.assertLess(len(digest), 2500)         # and the digest itself stays small


class ServerIntegrationTests(unittest.TestCase):
    def setUp(self):
        from local_chat import server as sv
        self.sv = sv
        sv._LAST_PLAN.clear(); sv._VERIFIED.clear(); sv._COORDINATOR.clear()
        sv._MODEL_INSPECTOR.invalidate()
        self.show = show_json()
        patches = [patch.object(sv, "_ollama_post", lambda path, payload, timeout=0: self.show),
                   patch.object(sv, "_ollama_tag_entries", lambda: [{"name": "m:27b", "size": 18410394087}]),
                   patch.object(sv, "_model_size", lambda m: 18410394087),
                   patch.object(hardware, "gpu_info", lambda *a, **k: gpu()),
                   patch.object(hardware, "ram_info", lambda: ram()),
                   patch.object(hardware, "all_gpus", lambda *a, **k: [gpu()])]
        for p in patches:
            p.start(); self.addCleanup(p.stop)

    def set_ps(self, models, reachable=True):
        state = ri.RuntimeState(reachable, tuple(ri.parse_ps({"models": models})), raw=tuple(models))
        p = patch.object(self.sv, "_loaded_ollama_models", lambda: models)
        p.start(); self.addCleanup(p.stop)
        q = patch.object(self.sv._RUNTIME, "_cached", state); q.start(); self.addCleanup(q.stop)

    def test_capacity_is_the_declared_architecture_limit(self):
        self.assertEqual(self.sv._model_context_capacity("m:27b"), 262144)

    def test_profile_uses_hybrid_formula_and_reports_plan(self):
        self.set_ps([])
        with patch.object(self.sv.app_settings, "runtime_settings", lambda: {"max_context": 0}):
            profile_ = self.sv._model_profile("m:27b")
        self.assertTrue(profile_["auto"])
        self.assertLess(profile_["num_ctx"], 262144)
        self.assertEqual(profile_["ctx_plan"]["method"], "hybrid_recurrent")
        self.assertEqual(profile_["ctx_decision"]["action"], "cold_load")

    def test_resident_window_is_kept_for_small_requests(self):
        self.set_ps([{"name": "m:27b", "size": 19 * GIB, "size_vram": 9567815596, "context_length": 8192}])
        p = self.sv._model_profile("m:27b", required_tokens=500)
        self.assertEqual(p["num_ctx"], 8192)
        self.assertFalse(p["ctx_decision"]["reload"])
        grown = self.sv._model_profile("m:27b", required_tokens=7000)
        self.assertTrue(grown["ctx_decision"]["reload"])
        self.assertGreater(grown["num_ctx"], 8192)

    def test_verification_reports_mismatch_and_never_fakes_success(self):
        self.set_ps([{"name": "m:27b", "size": 19 * GIB, "size_vram": 9 * GIB, "context_length": 8192}])
        result = self.sv._verify_runtime_context("m:27b", 32768, force=True)
        self.assertEqual(result["context"]["status"], "mismatch")
        self.assertFalse(result["context"]["verified"])
        self.assertNotIn("m:27b", self.sv._VERIFIED)
        ok = self.sv._verify_runtime_context("m:27b", 8192, force=True)
        self.assertTrue(ok["context"]["verified"])
        self.assertIsNone(self.sv._verify_runtime_context("m:27b", 8192))      # cached: no extra /api/ps work

    def test_verification_when_ollama_reports_no_context_is_unknown(self):
        self.set_ps([{"name": "m:27b", "size": 19 * GIB, "size_vram": 9 * GIB}])
        self.assertEqual(self.sv._verify_runtime_context("m:27b", 8192, force=True)["context"]["status"], "unknown")

    def test_unreachable_ollama(self):
        self.set_ps([], reachable=False)
        with patch.object(self.sv._RUNTIME, "_cached", ri.RuntimeState(False, ())):
            self.assertEqual(self.sv._verify_runtime_context("m:27b", 8192, force=True)["context"]["status"], "unreachable")

    def test_apply_context_reports_what_ollama_confirms(self):
        calls = []
        self.set_ps([{"name": "m:27b", "size": 19 * GIB, "size_vram": 9 * GIB, "context_length": 8192}])
        def fake_post(path, payload, timeout=0):
            calls.append((path, payload)); return {}
        with patch.object(self.sv, "_ollama_post", fake_post):
            result = self.sv._apply_context("m:27b", 32768)
        self.assertEqual(calls[0][0], "/api/generate")
        self.assertEqual(calls[0][1]["options"]["num_ctx"], 32768)
        self.assertFalse(result["applied"])           # /api/ps still says 8192 → not reported as success
        self.assertEqual(result["context"]["actual"], 8192)

    def test_growth_is_capped_by_the_hardware_plan(self):
        self.set_ps([])
        self.sv._model_profile("m:27b")
        plan = self.sv._LAST_PLAN["m:27b"]
        self.assertEqual(self.sv._growth_limit("m:27b"), min(262144, plan.max_feasible))

    def test_programming_mode_uses_the_adaptive_window(self):
        self.set_ps([])
        p = self.sv._programming_profile("m:27b")
        self.assertTrue(p["auto"])
        self.assertEqual(p["num_ctx"], p["ctx_plan"]["target"])

    def test_status_reports_declared_actual_and_plan(self):
        self.set_ps([{"name": "m:27b", "size": 19 * GIB, "size_vram": 9567815596, "context_length": 8192}])
        status = self.sv._runtime_status()
        item = status["contexts"][0]
        self.assertEqual((item["context"]["declared"], item["context"]["actual"]), (262144, 8192))
        self.assertEqual(item["profile"]["architecture"], "qwen35")
        self.assertTrue(item["plan"]["measured"]["ram"])


if __name__ == "__main__":
    unittest.main()
