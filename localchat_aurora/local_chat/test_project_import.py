"""Imported (user-supplied ZIP) projects: safe extraction, free layout, search, edit with backup."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from local_chat import system_tools as tools


def make_zip(entries, symlink=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
        if symlink:
            info = zipfile.ZipInfo(symlink[0])
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, symlink[1])
    return buffer.getvalue()


SESSION = "s" * 40
SITE = {
    "mysite/index.html": "<html><body><h1>سلام</h1><link rel=stylesheet href=css/style.css></body></html>",
    "mysite/css/style.css": "h1 { color: red; }\r\nbody { margin: 0; }\r\n",
    "mysite/js/app.js": "console.log('hi');\n",
    "mysite/img/logo.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
    "mysite/node_modules/lib/x.js": "junk",
    "mysite/.git/config": "secret",
    "mysite/.env": "KEY=1",
    "mysite/../evil.txt": "escape",
    "/abs/evil.txt": "abs",
    "mysite/data.json": '{"a": 1}',
}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(tools, "PROJECTS_ROOT", Path(self.tmp.name) / "projects")
        patcher.start()
        self.addCleanup(patcher.stop)

    def import_site(self, **kw):
        return tools.import_zip_project(make_zip(SITE, **kw), "My Site.zip", "att1")


class Import(Base):
    def test_extracts_safely_and_strips_top_folder(self):
        result = self.import_site(symlink=("mysite/link", "/etc/passwd"))
        self.assertEqual(result["slug"], "my-site")
        project = tools.PROJECTS_ROOT / "my-site"
        self.assertTrue((project / "index.html").is_file())
        self.assertTrue((project / "css" / "style.css").is_file())
        self.assertTrue((project / "img" / "logo.png").is_file())
        for bad in ("node_modules", ".git", ".env", "link"):
            self.assertFalse((project / bad).exists(), bad)
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())
        self.assertFalse((tools.PROJECTS_ROOT / "evil.txt").exists())
        self.assertTrue(tools.is_imported(project))
        self.assertTrue(result["skipped"])

    def test_same_attachment_reuses_project(self):
        first = self.import_site()
        (tools.PROJECTS_ROOT / first["slug"] / "js" / "app.js").write_text("changed", encoding="utf-8")
        second = tools.import_zip_project(make_zip(SITE), "My Site.zip", "att1")
        self.assertTrue(second["reused"])
        self.assertEqual(second["slug"], first["slug"])
        third = tools.import_zip_project(make_zip(SITE), "My Site.zip", "att2")
        self.assertNotEqual(third["slug"], first["slug"])

    def test_rejects_bad_archives(self):
        with self.assertRaises(tools.SystemToolError):
            tools.import_zip_project(b"not a zip", "a.zip", "x")
        with self.assertRaises(tools.SystemToolError):
            tools.import_zip_project(make_zip({"a/.env": "k"}), "a.zip", "x")
        self.assertFalse(list(tools.PROJECTS_ROOT.glob("*")) and any(p.name == "a" for p in tools.PROJECTS_ROOT.iterdir()))

    def test_persian_name_gets_ascii_slug(self):
        result = tools.import_zip_project(make_zip({"i.html": "<p>x</p>"}), "سایت من.zip", "p1")
        self.assertEqual(result["slug"], "site")


class Tools(Base):
    def setUp(self):
        super().setUp()
        self.slug = self.import_site()["slug"]

    def test_list_search_read(self):
        root = tools.read_only_action("list_files", {"project": self.slug})
        self.assertIn("css/", root["listing"])
        self.assertIn("index.html", root["listing"])
        sub = tools.read_only_action("list_files", {"project": self.slug, "path": "css"})
        self.assertIn("style.css", sub["listing"])
        found = tools.read_only_action("search_files", {"project": self.slug, "query": "color: red"})
        self.assertEqual(found["matches"], ["css/style.css:1: h1 { color: red; }"])
        read = tools.read_only_action("read_file", {"project": self.slug, "path": "index.html"})
        self.assertIn("سلام", read["content"])
        with self.assertRaises(tools.SystemToolError):
            tools.read_only_action("read_file", {"project": self.slug, "path": "img/logo.png"})
        with self.assertRaises(tools.SystemToolError):
            tools.read_only_action("read_file", {"project": self.slug, "path": "../evil.txt"})
        with self.assertRaises(tools.SystemToolError):
            tools.read_only_action("search_files", {"project": self.slug, "query": "x"})

    def test_edit_keeps_crlf_and_backs_up(self):
        plan = tools._plan("edit_file", {"project": self.slug, "path": "css/style.css", "old_text": "color: red", "new_text": "color: blue"})
        self.assertIn("blue", plan["diff"])
        token = tools.prepare_action("edit_file", {"project": self.slug, "path": "css/style.css", "old_text": "color: red",
                                                   "new_text": "color: blue"}, SESSION)["approval_id"]
        result = tools.approve_action(token, SESSION)
        self.assertTrue(result["ok"], result)
        data = (tools.PROJECTS_ROOT / self.slug / "css" / "style.css").read_bytes()
        self.assertIn(b"color: blue", data)
        self.assertIn(b"\r\n", data)
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
        self.assertTrue(list((tools.PROJECTS_ROOT / self.slug / ".history").glob("css__style.css.*.bak")))

    def test_new_file_anywhere_but_not_hidden_or_binary(self):
        tools._plan("write_file", {"project": self.slug, "path": "pages/about/index.html", "content": "<p>x</p>"})
        for bad in (".env", "a/.git/x.js", "logo2.png", "node_modules/x.js", "../x.html", "C:/x.html"):
            with self.assertRaises(tools.SystemToolError, msg=bad):
                tools._plan("write_file", {"project": self.slug, "path": bad, "content": "x"})

    def test_non_utf8_file_is_not_edited(self):
        path = tools.PROJECTS_ROOT / self.slug / "old.html"
        path.write_bytes("سلام".encode("cp1256"))
        with self.assertRaises(tools.SystemToolError):
            tools._plan("update_file", {"project": self.slug, "path": "old.html", "content": "x"})

    def test_check_project_and_context(self):
        (tools.PROJECTS_ROOT / self.slug / "data.json").write_text("{bad", encoding="utf-8")
        report = tools.read_only_action("check_project", {"project": self.slug})
        self.assertFalse(report["healthy"])
        self.assertIn("data.json", report["problems"][0])
        context = tools.project_context(self.slug)
        self.assertIn("css/ (1 فایل)", context)

    def test_manifest_survives_action_log(self):
        project = tools.PROJECTS_ROOT / self.slug
        tools._log_action(project, {"action": "edit_file", "path": "index.html"}, {"ok": True, "summary": "x"})
        self.assertTrue(tools.is_imported(project))


if __name__ == "__main__":
    unittest.main()
