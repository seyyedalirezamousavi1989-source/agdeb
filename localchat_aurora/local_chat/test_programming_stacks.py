"""Python / Node stacks, tool selection, context size and model choice."""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat import coding_agent, models_catalog, project_analysis, system_tools, toolchains, toolfinder

S1 = "a" * 40


class StackBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "projects"
        self.root.mkdir()
        self.patches = [patch.object(system_tools, "PROJECTS_ROOT", self.root)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_action(self, action, args):
        plan = system_tools.prepare_action(action, args, S1)
        return system_tools.approve_action(plan["approval_id"], S1)

    def calls(self, log):
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


class NodeTests(StackBase):
    def test_only_python_and_node_kinds_exist(self):
        self.assertEqual(set(toolchains.KINDS), {"python", "node"})
        self.run_action("create_project", {"name": "web1", "kind": "node"})
        self.assertEqual(system_tools._kind(self.root / "web1"), "node")
        for gone in ("flutter", "gradle"):
            with self.assertRaises(system_tools.SystemToolError):
                self.run_action("create_project", {"name": "x" + gone, "kind": gone})

    def test_static_site_files_are_allowed(self):
        self.run_action("create_project", {"name": "web1", "kind": "node"})
        for path, text in (("public/index.html", "<h1>x</h1>"), ("src/app.js", "1"), ("public/style.css", "a{}")):
            self.assertTrue(self.run_action("write_file", {"project": "web1", "path": path, "content": text})["ok"], path)
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("write_file", {"project": "web1", "path": "evil/x.js", "content": "x"})

    def test_task_and_package_validation(self):
        self.run_action("create_project", {"name": "web1", "kind": "node"})
        with patch.object(toolfinder, "find", lambda name: "/fake/" + name):   # validation happens before anything runs
            for task, extra in (("rm -rf", {}), ("add", {"packages": ["x; rm -rf /"]}), ("pub_get", {})):
                with self.assertRaises(system_tools.SystemToolError, msg=task):
                    self.run_action("toolchain_run", {"project": "web1", "task": task, **extra})

    def test_python_project_rejects_toolchain(self):
        self.run_action("create_project", {"name": "py1"})
        with self.assertRaises(system_tools.SystemToolError):
            self.run_action("toolchain_run", {"project": "py1", "task": "build"})


@unittest.skipUnless(shutil.which("npm") and shutil.which("node"), "node/npm not installed")
class NodeRealTests(StackBase):
    def test_node_init_and_test(self):
        with patch.object(toolfinder, "_configured", lambda: {}):
            self.run_action("create_project", {"name": "web1", "kind": "node"})
            self.assertTrue(self.run_action("toolchain_run", {"project": "web1", "task": "init"})["ok"])
            self.assertTrue((self.root / "web1/package.json").is_file())
            self.run_action("write_file", {"project": "web1", "path": "src/a.js", "content": "console.log(1)\n"})
            self.assertTrue(self.run_action("toolchain_run", {"project": "web1", "task": "test"})["ok"] in (True, False))


class ContextSizeTests(unittest.TestCase):
    def test_tools_are_compact_and_selected(self):
        full = len(json.dumps(coding_agent.TOOL_DEFINITIONS, ensure_ascii=False))
        small = len(json.dumps(coding_agent.select_tools("python", ["write_file"], False), ensure_ascii=False))
        self.assertLess(full, 6000)
        self.assertLess(small, full * 0.65)
        names = {t["function"]["name"] for t in coding_agent.select_tools("node", ["create_project"], False)}
        self.assertIn("toolchain_run", names)
        self.assertNotIn("venv_install", names)
        self.assertIn("create_project", names)
        for t in coding_agent.TOOL_DEFINITIONS:
            self.assertNotIn("purpose", t["function"]["parameters"]["properties"])

    def test_tool_message_is_trimmed(self):
        agent = coding_agent.CodingAgent.__new__(coding_agent.CodingAgent)
        agent.budget_chars = 12000
        text = agent._tool_message({"ok": True, "summary": "s", "action": "x", "path": "/abs", "import_graph": {"a": "b"}, "output": "o" * 20000})
        data = json.loads(text)
        self.assertNotIn("path", data)
        self.assertNotIn("import_graph", data)
        self.assertLess(len(text), 4000)

    def test_old_file_bodies_are_slimmed(self):
        big = "x" * 5000
        msgs = [{"role": "system", "content": "s"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "a", "content": big}}}]},
                {"role": "tool", "content": big}] + [{"role": "user", "content": "u"}] * 7
        slim = coding_agent.CodingAgent._slim(msgs)
        self.assertLess(len(json.dumps(slim)), 1500)


class SelectionTests(unittest.TestCase):
    def test_coding_model_choice(self):
        caps = {"llama3:8b": ["completion"], "qwen2.5-coder:7b": ["completion", "tools"], "qwen3:30b": ["completion", "tools"],
                "nomic": ["embedding"]}
        self.assertEqual(models_catalog.pick_coding_model(caps), "qwen2.5-coder:7b")
        self.assertEqual(models_catalog.pick_coding_model(caps, "qwen3:30b"), "qwen3:30b")
        self.assertIsNone(models_catalog.pick_coding_model({"a": ["completion"]}))

    def test_plan_has_kind(self):
        plan = coding_agent.parse_plan(json.dumps({"decision": "plan", "project": "app", "kind": "node", "stack_reason": "سایت",
                                                   "steps": [{"title": "t", "tool": "create_project"}]}))
        self.assertEqual(plan["kind"], "node")
        self.assertIn("node", coding_agent.plan_markdown(plan))
        self.assertEqual(coding_agent.parse_plan(json.dumps({"decision": "answer", "kind": "flutter"}))["kind"], "")
        self.assertEqual(coding_agent.parse_plan(json.dumps({"decision": "answer", "kind": "cobol"}))["kind"], "")
        self.assertNotIn("flutter", coding_agent.available_toolchains())
        self.assertIn("python=yes", coding_agent.available_toolchains())


if __name__ == "__main__":
    unittest.main()
