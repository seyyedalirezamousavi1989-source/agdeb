"""Programming mode with an imported project: read-only review, planned edits, no new-site design, usage observation."""
import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from local_chat import coding_agent as ca
from local_chat import server
from local_chat import system_tools as tools

SESSION = "s" * 40


def zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(tools, "PROJECTS_ROOT", Path(self.tmp.name) / "projects")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.slug = tools.import_zip_project(zip_bytes({"site/index.html": "<h1>Hi</h1>", "site/css/a.css": "h1{color:red}"}),
                                             "site.zip", "att")["slug"]

    def agent(self, text, imported=True, **kw):
        payload = {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": text}], "options": {}}
        events = []
        agent = ca.CodingAgent(emit=events.append, heartbeat=lambda: None, session_id=SESSION,
                               run={"cancelled": threading.Event()}, ollama_url="http://x", payload=payload, budget_chars=60000,
                               base_system="base", request_text=text, imported_project=self.slug if imported else "", **kw)
        agent.events = events
        return agent


class Routing(Base):
    def route(self, text, imported=True, **kw):
        agent = self.agent(text, imported, **kw)
        calls = []
        with patch.object(agent, "_review", lambda: calls.append("review")), \
                patch.object(agent, "_plan_and_run", lambda allow: calls.append("plan")), \
                patch.object(agent, "_answer", lambda phase: calls.append("answer:" + phase)):
            agent._turn()
        return calls, agent

    def test_analysis_is_read_only_review(self):
        for text in ("این پروژه رو تحلیل کن", "مشکلاتش چیه؟", "ساختارش رو توضیح بده"):
            self.assertEqual(self.route(text)[0], ["review"], text)

    def test_fix_request_plans_edits_of_same_project(self):
        calls, agent = self.route("منوی بالا رو راست‌چین کن و اصلاح کن")
        self.assertEqual(calls, ["plan"])
        self.assertTrue(agent.modify)

    def test_explicit_new_project_is_not_hijacked(self):
        self.assertNotIn("review", self.route("یک پروژه جدید بساز، ماشین حساب")[0])

    def test_new_site_from_scratch_is_not_offered(self):
        agent = self.agent("یک سایت فروشگاهی بساز", imported=False)
        with patch.object(ca.tools, "list_projects", return_value=[]), patch.object(ca.tools, "active_project_name", return_value=""):
            agent._turn()
        self.assertIn(ca.NO_SITE_TEXT, json.dumps(agent.events, ensure_ascii=False).replace("\\n", "\n"))
        self.assertEqual(ca.interview_questions("یک سایت بساز"), [])

    def test_fixing_an_existing_site_is_not_blocked(self):
        self.assertFalse(ca.wants_new_site("مشکل سایت رو درست کن"))
        self.assertFalse(ca.wants_new_site("این سایت رو اصلاح کن"))


class Review(Base):
    def test_loop_reads_files_and_blocks_writes(self):
        agent = self.agent("این پروژه رو تحلیل کن")
        seen = []
        script = iter([
            ("", [{"function": {"name": "list_files", "arguments": {"project": self.slug}}}], {}),
            ("", [{"function": {"name": "update_file", "arguments": {"project": self.slug, "path": "index.html", "content": "x"}}}], {}),
            ("", [{"function": {"name": "read_file", "arguments": {"project": self.slug, "path": "index.html"}}}], {}),
            ("نتیجه: h1 داخل index.html است.", [], {"done_reason": "stop"}),
        ])

        def fake_stream(payload, forward=True):
            seen.append([m for m in payload["messages"] if m["role"] == "tool"])
            return next(script)
        with patch.object(agent, "_stream", fake_stream):
            agent._review()
        tool_results = [json.loads(m["content"]) for m in seen[-1]]
        self.assertTrue(tool_results[0]["listing"].startswith("css/"))
        self.assertFalse(tool_results[1]["ok"])                      # write refused in review
        self.assertIn("<h1>Hi</h1>", tool_results[2]["content"])
        self.assertEqual((tools.PROJECTS_ROOT / self.slug / "index.html").read_text(encoding="utf-8"), "<h1>Hi</h1>")
        self.assertTrue(agent.events[-1]["done"])

    def test_imported_tools_have_no_venv_or_create_project(self):
        names = {d["function"]["name"] for d in ca.select_tools("python", ["create_project", "create_venv", "edit_file"], imported=True)}
        self.assertIn("edit_file", names)
        self.assertIn("search_files", names)
        self.assertNotIn("create_project", names)
        self.assertNotIn("create_venv", names)

    def test_py_path_is_not_rewritten_into_src(self):
        args = ca.normalize_arguments("read_file", {"project": self.slug, "path": "app.py"}, self.slug, keep_paths=True)
        self.assertEqual(args["path"], "app.py")
        args = ca.normalize_arguments("read_file", {"project": "p", "path": "app.py"}, "p")
        self.assertEqual(args["path"], "src/app.py")


class Observe(Base):
    def test_every_model_call_reports_usage(self):
        reported = []
        agent = self.agent("سلام", observe=lambda packet, payload: reported.append(packet))
        agent._observe({"done": True, "prompt_eval_count": 5}, {"messages": []})
        agent._observe({}, {})
        self.assertEqual(len(reported), 1)

    def test_observer_errors_never_break_the_turn(self):
        agent = self.agent("سلام", observe=lambda *a: 1 / 0)
        agent._observe({"done": True}, {})


class ServerImport(Base):
    def test_zip_attachment_becomes_project_and_is_reused(self):
        source = Path(self.tmp.name) / "a.source"
        source.write_bytes(zip_bytes({"x/index.html": "<p>a</p>"}))
        record = {"id": "r1", "kind": "archive", "name": "x.zip", "source_path": str(source)}
        imported, notices = server._import_archives_for_coding([[record]])
        slug = imported["r1"]["slug"]
        self.assertTrue((tools.PROJECTS_ROOT / slug / "index.html").is_file())
        self.assertTrue(notices)
        again, notices2 = server._import_archives_for_coding([[record]])
        self.assertEqual(again["r1"]["slug"], slug)
        self.assertEqual(notices2, [])

    def test_missing_original_falls_back_with_notice(self):
        imported, notices = server._import_archives_for_coding([[{"id": "r2", "kind": "archive", "name": "y.zip"}]])
        self.assertEqual(imported, {})
        self.assertIn("y.zip", notices[0])

    def test_corrupt_zip_creates_nothing(self):
        source = Path(self.tmp.name) / "b.source"
        source.write_bytes(b"not a zip")
        before = set(p.name for p in tools.PROJECTS_ROOT.iterdir())
        imported, notices = server._import_archives_for_coding([[{"id": "r3", "kind": "archive", "name": "z.zip", "source_path": str(source)}]])
        self.assertEqual(imported, {})
        self.assertEqual(before, set(p.name for p in tools.PROJECTS_ROOT.iterdir()))


if __name__ == "__main__":
    unittest.main()
