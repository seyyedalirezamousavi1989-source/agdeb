"""Memory / context visibility and automatic release (runtime_guard + settings + HTTP routes)."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from local_chat import runtime_guard, settings as app_settings

GB = 1024 ** 3


def _ps(name="qwen3:8b", size=6 * GB, vram=6 * GB, ctx=8192, minutes=4):
    expires = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.123456789Z")
    return {"name": name, "model": name, "size": size, "size_vram": vram, "context_length": ctx, "expires_at": expires}


class SummaryTests(unittest.TestCase):
    def test_nanosecond_timestamps_and_gpu_share(self):
        model = runtime_guard.summarize_loaded([_ps(vram=3 * GB)])[0]
        self.assertEqual(model["gpu_percent"], 50)
        self.assertEqual(model["context"], 8192)
        self.assertTrue(200 <= model["expires_in"] <= 241)

    def test_garbage_entries_are_ignored(self):
        self.assertEqual(runtime_guard.summarize_loaded([None, {}, "x", {"name": ""}]), [])

    def test_findings_name_the_cause(self):
        models = runtime_guard.summarize_loaded([_ps("a"), _ps("b", ctx=32768, vram=2 * GB)])
        text = " ".join(f["text"] for f in runtime_guard.diagnose(models, task_reason="x"))
        for expected in ("کار طولانی", "2 مدل", "GPU", "32768"):
            self.assertIn(expected, text)
        text = " ".join(f["text"] for f in runtime_guard.diagnose(models, auto_resume=True))
        self.assertIn("ازسرگیری خودکار", text)
        self.assertIn("آزاد", runtime_guard.diagnose([])[0]["text"])


class GuardianTests(unittest.TestCase):
    def _guardian(self, loaded, busy, minutes=5):
        clock = {"t": 0.0}
        released = []
        guard = runtime_guard.IdleGuardian(lambda: list(loaded), lambda n: released.append(n) or True,
                                           lambda: busy["v"], lambda: minutes, clock=lambda: clock["t"])
        return guard, clock, released

    def test_unloads_only_after_idle_period(self):
        busy = {"v": False}
        guard, clock, released = self._guardian([{"name": "m1"}, {"name": "m2"}], busy)
        self.assertEqual(guard.tick(), [])
        clock["t"] = 299
        self.assertEqual(guard.tick(), [])
        clock["t"] = 301
        self.assertEqual(guard.tick(), ["m1", "m2"])
        self.assertEqual(guard.last_release["reason"], "idle")

    def test_work_resets_the_idle_clock(self):
        busy = {"v": False}
        guard, clock, released = self._guardian([{"name": "m1"}], busy)
        guard.tick()
        clock["t"] = 280
        busy["v"] = True
        guard.tick()
        busy["v"] = False
        clock["t"] = 400
        self.assertEqual(guard.tick(), [])           # idle clock restarted at t=400
        clock["t"] = 701
        self.assertEqual(guard.tick(), ["m1"])

    def test_zero_minutes_disables_it(self):
        busy = {"v": False}
        guard, clock, released = self._guardian([{"name": "m1"}], busy, minutes=0)
        guard.tick()
        clock["t"] = 10_000
        self.assertEqual(guard.tick(), [])
        self.assertEqual(released, [])

    def test_failed_unload_does_not_stop_the_others(self):
        guard = runtime_guard.IdleGuardian(lambda: [{"name": "bad"}, {"name": "good"}],
                                           lambda n: (_ for _ in ()).throw(OSError()) if n == "bad" else True,
                                           lambda: False, lambda: 1)
        self.assertEqual(guard.release_all("manual"), ["good"])


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_defaults_are_safe(self):
        self.assertEqual(app_settings.runtime_settings(), {"auto_resume_tasks": False, "idle_unload_minutes": 5, "max_context": 32768})

    def test_round_trip_and_validation(self):
        self.assertEqual(app_settings.set_runtime_settings({"auto_resume_tasks": True, "idle_unload_minutes": 0, "max_context": 32768}),
                         {"auto_resume_tasks": True, "idle_unload_minutes": 0, "max_context": 32768})
        for bad in ({"auto_resume_tasks": "yes"}, {"idle_unload_minutes": -1}, {"idle_unload_minutes": 9999},
                    {"idle_unload_minutes": True}, {"idle_unload_minutes": "5"}):
            with self.assertRaises(ValueError):
                app_settings.set_runtime_settings(bad)


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------------------------------ HTTP routes
from local_chat import server as srv  # noqa: E402
from local_chat.test_features import FeatureTestCase  # noqa: E402


class RuntimeRoutesTests(FeatureTestCase):
    def setUp(self):
        super().setUp()
        self.released = []
        self.loaded = [_ps("chat:8b"), _ps("embed:1b", size=GB, vram=GB, ctx=2048)]
        patches = [
            patch.object(srv, "_loaded_ollama_models", lambda: list(self.loaded)),
            patch.object(srv, "_release_ollama_model", lambda name: self.released.append(name) or True),
            patch.object(srv, "_busy_task_reason", lambda: ""),
            patch.object(srv, "RUNTIME_GUARD", None),
            patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": self.tmp.name}),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(lambda: srv.RUNTIME_GUARD and srv.RUNTIME_GUARD.stop())

    def test_status_lists_models_and_reasons(self):
        status, body = self.call("GET", "/api/runtime/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual([m["name"] for m in data["models"]], ["chat:8b", "embed:1b"])
        self.assertEqual(data["settings"], {"auto_resume_tasks": False, "idle_unload_minutes": 5, "max_context": 32768})
        self.assertTrue(any("2 مدل" in f["text"] for f in data["findings"]))

    def test_release_unloads_everything_when_idle(self):
        status, body = self.call("POST", "/api/runtime/release", {})
        self.assertEqual(status, 200)
        self.assertEqual(self.released, ["chat:8b", "embed:1b"])
        self.assertEqual(json.loads(body)["last_release"]["reason"], "manual")

    def test_release_is_refused_while_a_task_runs(self):
        with patch.object(srv, "_busy_task_reason", lambda: "ترجمه در جریان است"):
            status, body = self.call("POST", "/api/runtime/release", {})
        self.assertEqual(status, 409)
        self.assertEqual(self.released, [])

    def test_settings_validate_and_persist(self):
        status, body = self.call("POST", "/api/runtime/settings", {"idle_unload_minutes": 2, "auto_resume_tasks": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["settings"], {"auto_resume_tasks": True, "idle_unload_minutes": 2, "max_context": 32768})
        self.assertEqual(self.call("POST", "/api/runtime/settings", {"idle_unload_minutes": -5})[0], 400)

    def test_panel_script_is_served_and_linked(self):
        status, body = self.call("GET", "/static/runtime_panel.js")
        self.assertEqual(status, 200)
        self.assertIn(b"/api/runtime/status", body)
        self.assertIn(b"runtime_panel.js", self.call("GET", "/")[1])


class MakeRoomTests(unittest.TestCase):
    """The reported failure: «حدود 4058 توکن از 4096» on a big model (4096 window, 2048 reserved for the answer)."""

    def _messages(self, system_chars):
        return [{"role": "system", "content": "ا" * system_chars}, {"role": "user", "content": "سلام، یک تحلیل کوتاه بده"}]

    def test_fitting_request_is_left_alone(self):
        from local_chat import context_manager as cm
        ctx, predict, ok, _ = cm.make_room(self._messages(1000), 4096, 2048)
        self.assertEqual((ctx, predict, ok), (4096, 2048, True))

    def test_output_reserve_shrinks_before_refusing(self):
        from local_chat import context_manager as cm
        messages = self._messages(4400)                       # ~2000 input tokens: fails with a 2048 reserve
        self.assertFalse(cm.guard_request(messages, 4096, 2048)[0])
        ctx, predict, ok, budget = cm.make_room(messages, 4096, 2048)
        self.assertTrue(ok)
        self.assertEqual(ctx, 4096)
        self.assertTrue(cm.MIN_PREDICT <= predict < 2048)
        self.assertTrue(budget.fits)

    def test_window_grows_when_shrinking_is_not_enough(self):
        from local_chat import context_manager as cm
        messages = self._messages(6500)                       # ~3000 input tokens
        ctx, predict, ok, _ = cm.make_room(messages, 4096, 2048, capacity=32768)
        self.assertTrue(ok)
        self.assertGreater(ctx, 4096)
        self.assertGreaterEqual(predict, cm.MIN_PREDICT)

    def test_window_never_exceeds_model_capacity(self):
        from local_chat import context_manager as cm
        ctx, predict, ok, _ = cm.make_room(self._messages(6500), 4096, 2048, capacity=4096)
        self.assertFalse(ok)
        self.assertEqual(ctx, 4096)
