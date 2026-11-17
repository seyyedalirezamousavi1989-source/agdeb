"""Safety tests for automatic restart recovery of persisted document tasks."""
import tempfile
import threading
import unittest
from pathlib import Path

from local_chat import task_service, tasks


class AutoRecoveryTests(unittest.TestCase):
    def _interrupted_task(self, root):
        manager = tasks.TaskManager(root)
        task_id = manager.create("summarize", "source.txt", b"A short document for recovery testing.\n" * 30,
                                 {"model": "test-model:latest", "target": "fa", "num_ctx": 4096,
                                  "num_predict": 512, "max_tokens": 300, "workers": 1})
        manager.prepare(task_id)
        manager._set(task_id, status=tasks.INTERRUPTED, stage="interrupted", owner="", run_started=None)
        return task_id

    def test_missing_model_leaves_task_interrupted(self):
        with tempfile.TemporaryDirectory() as root:
            task_id = self._interrupted_task(root)
            service = task_service.TaskService(lambda: root, lambda *a, **k: {}, lambda m: 4096,
                                               model_available=lambda model: False)
            manager = service.manager
            self.assertEqual(manager.get(task_id)["status"], tasks.INTERRUPTED)
            self.assertNotIn(task_id, manager._runners)

    def test_installed_model_starts_only_interrupted_task(self):
        with tempfile.TemporaryDirectory() as root:
            task_id = self._interrupted_task(root)
            gate = threading.Event()                       # keeps the worker busy so the runner is still registered when we look

            def slow_post(*args, **kwargs):
                gate.wait(5)
                return {"message": {"content": "summary"}}
            service = task_service.TaskService(lambda: root, slow_post,
                                               lambda m: 4096, model_available=lambda model: model == "test-model:latest",
                                               auto_resume=lambda: True)
            try:
                manager = service.manager
                self.assertIn(task_id, manager._runners)
                self.assertNotEqual(manager.get(task_id)["status"], tasks.INTERRUPTED)
            finally:
                gate.set()
                service.stop_all(2)

    def test_restart_does_not_resume_by_default(self):
        """The reported bug: a restart silently re-ran the half-finished translation and re-occupied the model."""
        with tempfile.TemporaryDirectory() as root:
            task_id = self._interrupted_task(root)
            calls = []
            service = task_service.TaskService(lambda: root, lambda *a, **k: calls.append(a) or {},
                                               lambda m: 4096, model_available=lambda model: True)
            manager = service.manager
            self.assertEqual(manager.get(task_id)["status"], tasks.INTERRUPTED)
            self.assertNotIn(task_id, manager._runners)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
