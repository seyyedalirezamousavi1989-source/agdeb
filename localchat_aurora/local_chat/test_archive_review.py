"""Programming mode with an attached project ZIP: review it, never plan a new project + venv for it."""
import unittest
from unittest.mock import patch

from local_chat import coding_agent as ca


def make_agent(text, attached):
    payload = {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": text}], "options": {}}
    return ca.CodingAgent(emit=lambda e: None, heartbeat=lambda: None, session_id="s", run={"cancelled": __import__("threading").Event()},
                          ollama_url="http://x", payload=payload, budget_chars=9000, base_system="base",
                          request_text=text, attached_archive=attached)


class ArchiveReview(unittest.TestCase):
    def run_turn(self, text, attached):
        agent = make_agent(text, attached)
        calls = []
        with patch.object(ca.tools, "list_projects", return_value=[]), \
                patch.object(ca.tools, "active_project_name", return_value=""), \
                patch.object(agent, "_answer", lambda phase: calls.append(("answer", phase))), \
                patch.object(agent, "_make_plan", lambda allow: calls.append(("plan",)) or None), \
                patch.object(agent, "_done", lambda *a, **k: calls.append(("done",))):
            agent._turn()
        return calls

    def test_fix_request_with_zip_is_reviewed_not_planned(self):
        for text in ("اصلاح کن", "مشکلاتش رو پیدا کن و درست کن", "این سایت رو بهتر کن"):
            self.assertEqual(self.run_turn(text, True), [("answer", "archive_review")], text)

    def test_explicit_new_project_still_plans(self):
        calls = self.run_turn("یک پروژه جدید بساز، بازی مار", True)
        self.assertNotIn(("answer", "archive_review"), calls)

    def test_without_zip_nothing_changes(self):
        self.assertNotIn(("answer", "archive_review"), self.run_turn("اصلاح کن", False))

    def test_prompt_forbids_tools_and_venv(self):
        text = ca.build_system_prompt("b", "archive_review")
        self.assertIn("هیچ ابزاری نداری", text)
        self.assertIn("محیط مجازی", text)


if __name__ == "__main__":
    unittest.main()
