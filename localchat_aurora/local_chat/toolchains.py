"""Fixed, allow-listed command templates for the Node.js stack (npm).

Nothing here runs a command; it only *builds* argument lists (never a shell string) from validated inputs.
The model chooses a ``task`` from a short list and a few validated parameters; it never writes a command line.
"""
import json
import os
import re
from pathlib import Path

if __package__:
    from . import toolfinder
else:
    import toolfinder

WINDOWS = os.name == "nt"


class ToolchainError(Exception):
    pass


# ----------------------------------------------------------------------------- project kinds
COMMON_EXT = {".txt", ".md", ".json", ".csv", ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".toml", ".yaml", ".yml",
              ".ini", ".cfg", ".xml", ".sql", ".svg"}
KINDS = {
    "python": {"label": "Python", "dirs": ("src", "tests", "data", "assets", "docs"),
               "root": {"requirements.txt", "README.md", ".gitignore"}, "ext": COMMON_EXT | {".py"}},
    "node": {"label": "Node.js", "dirs": ("src", "test", "public", "docs"),
             "root": {"package.json", "README.md", ".gitignore", "index.js", "index.html", "tsconfig.json", "vite.config.js"},
             "ext": COMMON_EXT | {".mjs", ".cjs", ".vue", ".scss"}},
}
KIND_TOOL = {"node": "npm"}


def policy(kind):
    return KINDS.get(kind) or KINDS["python"]


# ----------------------------------------------------------------------------- validation
NPM_PACKAGE = re.compile(r"^(@[a-z0-9][a-z0-9._-]{0,60}/)?[a-z0-9][a-z0-9._-]{0,60}(@[0-9A-Za-z.^~<>=*-]{1,20})?$")
TASKS = {
    "node": {"init", "install", "add", "test", "build", "start"},
}
TASK_HELP = {
    "node": "init, install, add (packages), test, build, start",
}

def _tool_env(*executables):
    path_parts = [str(Path(exe).parent) for exe in executables if exe]
    git = toolfinder.find("git")
    if git:
        path_parts.append(str(Path(git).parent))
    # Preserve the host environment (SystemRoot, TEMP, USERPROFILE, APPDATA,
    # COMSPEC, etc.). A minimal environment can make .bat/.cmd toolchains fail
    # on Windows even when the executable is correctly discovered.
    env = dict(os.environ)
    env.update({"PATH": os.pathsep.join(path_parts + [os.environ.get("PATH", "")]), "CI": "true",
                "NPM_CONFIG_UPDATE_NOTIFIER": "false"})
    return env


def _packages(value, pattern, label):
    if not isinstance(value, list) or not 1 <= len(value) <= 6 or any(not isinstance(item, str) or not pattern.fullmatch(item) for item in value):
        raise ToolchainError(f"packages باید فهرست ۱ تا ۶ نام معتبر {label} باشد.")
    return list(value)


def build(project, kind, task, args):
    """Return {argv, cwd, timeout, env, text, tool, ok_on_timeout} for an allow-listed task or raise ToolchainError."""
    if kind not in KIND_TOOL:
        raise ToolchainError("این پروژه از نوع Python است؛ برای آن از venv_* استفاده کن. toolchain_run فقط برای پروژهٔ Node.js است.")
    if task not in TASKS[kind]:
        raise ToolchainError(f"task مجاز برای {kind}: {TASK_HELP[kind]}")
    slug = project.name
    spec = {"cwd": str(project), "ok_on_timeout": False, "tool": KIND_TOOL[kind]}
    # Node.js / npm. On Windows prefer invoking npm's JavaScript CLI through
    # node.exe directly. This avoids cmd.exe/PATHEXT quoting differences between
    # Windows versions and also works when npm.cmd is discovered outside PATH.
    exe = toolfinder.find("npm")
    if not exe:
        raise ToolchainError("npm پیدا نشد (نه در PATH و نه در پوشه‌های معمول). Node.js را نصب کن یا مسیر را در tool_paths.json بنویس.")
    npm_prefix = [exe]
    if WINDOWS and Path(exe).suffix.lower() in {".cmd", ".bat"}:
        node_exe = toolfinder.find("node")
        npm_cli_candidates = [
            Path(exe).parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
            Path(exe).parent / "node_modules" / "npm" / "bin" / "npm.js",
        ]
        npm_cli = next((candidate for candidate in npm_cli_candidates if candidate.is_file()), None)
        if node_exe and npm_cli:
            npm_prefix = [node_exe, str(npm_cli)]
    base_install = ["install", "--ignore-scripts", "--no-audit", "--no-fund"]
    timeout = 600
    if task == "init":
        if (project / "package.json").exists():
            raise ToolchainError("package.json از قبل وجود دارد.")
        argv, text = [*npm_prefix, "init", "-y"], "ساخت package.json"
    elif task == "install":
        argv, text = [*npm_prefix, *base_install], "نصب وابستگی‌ها (npm install، بدون اسکریپت‌های نصب)"
    elif task == "add":
        pkgs = _packages(args.get("packages"), NPM_PACKAGE, "npm")
        argv, text = [*npm_prefix, *base_install, *pkgs], "افزودن بسته‌ها: " + ", ".join(pkgs)
    elif task == "test":
        argv, text, timeout = [*npm_prefix, "test"], "اجرای تست‌ها (npm test؛ اسکریپت test پروژه اجرا می‌شود)", 600
    elif task == "build":
        argv, text, timeout = [*npm_prefix, "run", "build"], "ساخت (npm run build)", 900
    else:
        argv, text, timeout = [*npm_prefix, "start"], "اجرای برنامه (npm start، حداکثر ۳۰ ثانیه)", 30
        spec["ok_on_timeout"] = True
    if task != "init" and not (project / "package.json").exists():
        raise ToolchainError("package.json وجود ندارد؛ ابتدا toolchain_run با task=init را اجرا کن.")
    return {**spec, "argv": argv, "timeout": timeout, "env": _tool_env(*npm_prefix), "text": text}
