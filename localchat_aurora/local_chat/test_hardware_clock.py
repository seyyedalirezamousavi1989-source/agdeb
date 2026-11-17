"""Hardware-aware window, internal clock, lean per-turn prompt, forex history compaction."""
import json
import os
import tempfile
import time
import unittest
from email.utils import formatdate
from types import SimpleNamespace
from unittest.mock import patch

from local_chat import clock, hardware, server as srv, settings as app_settings
from local_chat.test_features import FeatureTestCase

GB = 1024 ** 3
QWEN14 = {"model_info": {"general.architecture": "qwen2", "qwen2.block_count": 40, "qwen2.attention.head_count": 40,
                         "qwen2.attention.head_count_kv": 8, "qwen2.embedding_length": 5120, "qwen2.context_length": 32768}}


def gpu(total_gb, used_gb=0.5):
    return {"name": "Test GPU", "total": int(total_gb * GB), "used": int(used_gb * GB), "free": int((total_gb - used_gb) * GB)}


class WindowTests(unittest.TestCase):
    def test_kv_cost_from_model_info(self):
        self.assertEqual(hardware.kv_bytes_per_token(QWEN14), 40 * 8 * (128 + 128) * 2)
        self.assertIsNone(hardware.kv_bytes_per_token({}))
        self.assertIsNone(hardware.kv_bytes_per_token({"model_info": {"general.architecture": "x"}}))

    def test_small_gpu_keeps_a_small_window_and_says_it_spills(self):
        plan = hardware.pick_window(9 * GB, 163840, 32768, gpu(12), other_vram_bytes=int(0.5 * GB))
        self.assertEqual(plan["num_ctx"], 8192)
        self.assertFalse(plan["fits_gpu"])

    def test_big_gpu_grows_up_to_the_user_ceiling(self):
        plan = hardware.pick_window(9 * GB, 163840, 32768, gpu(24), int(0.5 * GB))
        self.assertEqual((plan["num_ctx"], plan["fits_gpu"]), (32768, True))              # new default ceiling
        plan = hardware.pick_window(9 * GB, 163840, 32768, gpu(24), int(0.5 * GB), max_context=32768)
        self.assertEqual(plan["num_ctx"], 32768)

    def test_mid_gpu_picks_what_fits_and_never_exceeds_model_capacity(self):
        plan = hardware.pick_window(9 * GB, 163840, 32768, gpu(16), int(0.5 * GB), max_context=32768)
        self.assertTrue(8192 <= plan["num_ctx"] <= 24576)
        used = hardware.OVERHEAD_BYTES + 9 * GB + plan["num_ctx"] * 163840 * hardware.KV_SAFETY + int(0.5 * GB)
        self.assertLessEqual(used, 16 * GB * hardware.VRAM_USABLE)
        capped = hardware.pick_window(9 * GB, 163840, 8192, gpu(24), 0, max_context=32768)
        self.assertEqual(capped["num_ctx"], 8192)

    def test_no_gpu_data_means_no_opinion_without_ram_either(self):
        self.assertIsNone(hardware.pick_window(9 * GB, 163840, 32768, None))

    def test_hybrid_gpu_and_ram_can_exceed_the_old_4096_fallback(self):
        plan = hardware.pick_window(16 * GB, 320000, 65536, gpu(12, 1), 0,
                                   max_context=32768,
                                   ram={"total": 48 * GB, "available": 32 * GB})
        self.assertGreater(plan["num_ctx"], 4096)
        self.assertLessEqual(plan["num_ctx"], 32768)
        self.assertGreater(plan["ram_kv_mb"], 0)

    def test_ram_only_planning_is_bounded_by_available_memory(self):
        plan = hardware.pick_window(8 * GB, 160000, 32768, None, max_context=32768,
                                   ram={"total": 48 * GB, "available": 20 * GB})
        self.assertGreaterEqual(plan["num_ctx"], 4096)
        self.assertLessEqual(plan["num_ctx"], 32768)

    def test_never_exceeds_a_models_declared_small_context(self):
        plan = hardware.pick_window(4 * GB, 160000, 2048, gpu(24), max_context=32768,
                                   ram={"total": 48 * GB, "available": 32 * GB})
        self.assertEqual(plan["num_ctx"], 2048)

    def test_nvidia_smi_output_is_parsed(self):
        out = SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 4070 Ti, 12282, 1500, 10500\n")
        with patch.object(hardware, "_nvidia_smi_path", return_value="nvidia-smi"):
            gpus = hardware._query_gpus(lambda *a, **k: out)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4070 Ti")
        self.assertEqual(gpus[0]["total"], 12282 * 1024 ** 2)


class ProfileTests(unittest.TestCase):
    def _patches(self, gpu_data):
        return [patch.object(hardware, "gpu_info", return_value=gpu_data),
                patch.object(srv, "_model_size", return_value=9 * GB),
                patch.object(srv, "_model_context_capacity", return_value=32768),
                patch.object(srv, "_loaded_ollama_models", return_value=[]),
                patch.object(srv, "_KV_CACHE", {"m:14b": 163840})]

    def test_profile_follows_the_gpu(self):
        for item in self._patches(gpu(24)):
            item.start(); self.addCleanup(item.stop)
        profile = srv._model_profile("m:14b")
        self.assertTrue(profile["auto"])
        self.assertEqual(profile["num_ctx"], 32768)
        self.assertGreater(profile["budget"], 20000)
        forex = srv._forex_profile(profile)
        self.assertEqual(forex["num_ctx"], 32768)                  # preserves the larger adaptive window
        self.assertGreaterEqual(forex["num_predict"], profile["num_predict"])

    def test_without_gpu_the_old_size_rule_still_applies(self):
        for item in self._patches(None):
            item.start(); self.addCleanup(item.stop)
        self.assertEqual(srv._model_profile("m:14b")["num_ctx"], 4096)


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        clock.reset_for_tests()
        self.addCleanup(clock.reset_for_tests)

    @staticmethod
    def _opener(seconds_ahead):
        def opener(request, timeout=0):
            header = formatdate(time.time() + seconds_ahead, usegmt=True)
            return SimpleNamespace(headers={"Date": header}, __enter__=lambda s: s, __exit__=lambda *a: None)
        class Ctx:
            def __init__(self, response): self.response = response
            def __enter__(self): return self.response
            def __exit__(self, *a): return None
        return lambda request, timeout=0: Ctx(opener(request, timeout))

    def test_offline_by_default_and_sync_stores_a_correction(self):
        self.assertFalse(clock.status()["synced"])
        before = time.time()
        self.assertAlmostEqual(clock.epoch(), before, delta=2)
        info = clock.sync(self._opener(3600), hosts=("https://example.test",))
        self.assertTrue(info["synced"])
        self.assertAlmostEqual(info["offset_seconds"], 3600, delta=3)
        self.assertAlmostEqual(clock.epoch(), time.time() + 3600, delta=3)
        clock.reset_for_tests()                                    # a restart: the correction comes from the settings file
        self.assertAlmostEqual(clock.epoch(), time.time() + 3600, delta=3)

    def test_tiny_difference_is_not_carried(self):
        clock.sync(self._opener(0.4), hosts=("https://example.test",))
        self.assertEqual(clock.status()["offset_seconds"], 0.0)

    def test_failure_raises_and_keeps_the_clock_usable(self):
        def broken(request, timeout=0):
            raise OSError("offline")
        with self.assertRaises(OSError):
            clock.sync(broken, hosts=("https://a.test", "https://b.test"))
        self.assertAlmostEqual(clock.epoch(), time.time(), delta=2)

    def test_maybe_sync_skips_a_fresh_correction(self):
        clock.sync(self._opener(5), hosts=("https://example.test",))
        self.assertFalse(clock.maybe_sync(self._opener(5)))


class LeanPromptTests(unittest.TestCase):
    def test_general_turn_without_cues_is_lean_and_points_to_the_web_tab(self):
        lean = srv._system_prompt(time_cue=False, identity_cue=False, web_sources=False)
        full = srv._system_prompt()
        self.assertNotIn("زمان فعلی دستگاه کاربر", lean)
        self.assertNotIn("هویت ثابت برنامه", lean)
        self.assertIn("تب «وب»", lean)
        self.assertLess(len(lean), len(full))

    def test_cues(self):
        self.assertTrue(srv._time_cue("الان ساعت چنده؟"))
        self.assertTrue(srv._time_cue("what time is it"))
        self.assertFalse(srv._time_cue("یک تابع پایتون بنویس"))
        self.assertTrue(srv._identity_cue("کی تو رو ساخته؟"))
        self.assertFalse(srv._identity_cue("سلام، یک شعر بگو"))
        self.assertTrue(srv._needs_live_info("قیمت لحظه‌ای طلا چنده"))
        self.assertTrue(srv._needs_live_info("latest news about bitcoin"))
        self.assertFalse(srv._needs_live_info("تفاوت لیست و تاپل چیست"))

    def test_web_sources_are_detected_and_change_the_wording(self):
        messages = [{"role": "system", "content": "منابع…\nشناسهٔ پیام جست‌وجو: abc"}, {"role": "user", "content": "x"}]
        self.assertTrue(srv._has_web_sources(messages))
        self.assertFalse(srv._has_web_sources([{"role": "user", "content": "x"}]))
        self.assertIn("منابع وب همین درخواست پیوست شده‌اند", srv._system_prompt(web_sources=True))


class ForexHistoryTests(unittest.TestCase):
    def test_only_the_newest_analysis_stays_whole(self):
        long = "تحلیل " * 400
        history = [{"role": "user", "content": "xauusd"}, {"role": "assistant", "content": long},
                   {"role": "user", "content": "h1"}, {"role": "assistant", "content": long},
                   {"role": "user", "content": "بازار؟"}]
        out = srv._compact_old_forex_turns(history)
        self.assertLess(len(out[1]["content"]), 1000)
        self.assertIn("کهنه", out[1]["content"])
        self.assertEqual(out[3]["content"], long)                  # the newest assistant turn is untouched
        self.assertEqual([m["content"] for m in out if m["role"] == "user"], ["xauusd", "h1", "بازار؟"])

    def test_chart_turns_and_short_turns_are_left_alone(self):
        history = [{"role": "assistant", "content": "x" * 5000, "chart": {"symbol": "XAUUSD"}},
                   {"role": "assistant", "content": "کوتاه"}, {"role": "assistant", "content": "y" * 3000}]
        out = srv._compact_old_forex_turns(history)
        self.assertEqual(len(out[0]["content"]), 5000)
        self.assertEqual(out[1]["content"], "کوتاه")


class RoutesTests(FeatureTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        clock.reset_for_tests()
        self.addCleanup(clock.reset_for_tests)

    def test_time_reports_the_internal_clock(self):
        status, body = self.call("GET", "/api/time")
        self.assertEqual(status, 200)
        self.assertIn("clock", json.loads(body))

    def test_sync_route_success_and_offline_failure(self):
        with patch.object(srv.app_clock, "sync", return_value={"synced": True, "offset_seconds": 1.0}):
            status, body = self.call("POST", "/api/time/sync", {})
        self.assertEqual(status, 200)
        with patch.object(srv.app_clock, "sync", side_effect=OSError("offline")):
            status, body = self.call("POST", "/api/time/sync", {})
        self.assertEqual(status, 502)

    def test_max_context_setting_is_validated(self):
        self.assertEqual(self.call("POST", "/api/runtime/settings", {"max_context": 24576})[0], 200)
        self.assertEqual(app_settings.runtime_settings()["max_context"], 24576)
        self.assertEqual(self.call("POST", "/api/runtime/settings", {"max_context": 5000})[0], 400)


if __name__ == "__main__":
    unittest.main()


class FollowupGuardTests(unittest.TestCase):
    """The analysis-writing call used to skip the token guard entirely."""

    def test_oversized_followup_gets_room_instead_of_a_silent_cut(self):
        profile = {"num_ctx": 4096, "num_predict": 2048, "budget": 4300}
        messages = [{"role": "system", "content": "ا" * 4400}, {"role": "user", "content": "تحلیل"}]
        request = {"model": "m", "messages": messages, "options": {"num_ctx": 4096, "num_predict": 2048}}
        with patch.object(srv, "_model_context_capacity", return_value=32768):
            fits, usage, notice = srv._guard_followup(request, lambda budget: messages, profile, "m")
        self.assertTrue(fits)
        self.assertTrue(notice)
        self.assertTrue(request["options"]["num_predict"] < 2048 or request["options"]["num_ctx"] > 4096)

    def test_small_followup_is_untouched(self):
        profile = {"num_ctx": 8192, "num_predict": 2560, "budget": 11000}
        messages = [{"role": "system", "content": "کوتاه"}, {"role": "user", "content": "تحلیل"}]
        request = {"model": "m", "messages": messages, "options": {"num_ctx": 8192, "num_predict": 2560}}
        fits, usage, notice = srv._guard_followup(request, lambda budget: messages, profile, "m")
        self.assertEqual((fits, notice), (True, ""))
        self.assertEqual(request["options"], {"num_ctx": 8192, "num_predict": 2560})
