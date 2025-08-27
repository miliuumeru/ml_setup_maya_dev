from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ---------- Constants ----------
DIRS = [
    "config",
    "logs",
    "cache",
    "exports",
    "temp",
    "scripts",
    "modules",
    "site-packages",
    "icons",
    "plug-ins",
    "tests",
    ".vscode",
]

GITIGNORE = """\
# --- Project-specific ---
logs/
cache/
exports/
temp/
site-packages/
*.log

# --- Python ---
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
.eggs/
.cache/
.mypy_cache/
.pytype/

# --- Virtual envs ---
.venv/
venv/
ENV/
env/

# --- VSCode noise (keep settings.json tracked) ---
.vscode/*.log
.vscode/*.db
.vscode/*.cache

# --- OS noise ---
.DS_Store
Thumbs.db

# --- Build/Dist ---
build/
dist/
wheels/
*.egg
*.spec

# --- Tests/Coverage ---
htmlcov/
.tox/
.nox/
.coverage*
.pytest_cache/
"""

VSCODE_SETTINGS = lambda pkg: {
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        r"${workspaceFolder}/scripts",
        r"${workspaceFolder}/site-packages"
    ],
    "python.analysis.indexing": True,
    "python.analysis.typeCheckingMode": "basic",
    "python.envFile": r"${workspaceFolder}/.env",
    "python.defaultInterpreterPath": r"${workspaceFolder}/.venv/Scripts/python.exe"
        if platform.system().lower().startswith("win")
        else r"${workspaceFolder}/.venv/bin/python",
    "python.testing.pytestEnabled": True,
    "python.testing.unittestEnabled": False,
    "python.testing.pytestArgs": ["-q", "tests"],
}

VSCODE_TASKS = {
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ruff: lint",
            "type": "shell",
            "command": r"${workspaceFolder}/.venv/Scripts/python.exe -m ruff ."
                if platform.system().lower().startswith("win")
                else r"${workspaceFolder}/.venv/bin/python -m ruff .",
            "problemMatcher": []
        }
    ]
}

README_TMPL = lambda name: f"""\
# {name}

Opinionated scaffold for Maya rig/animation tooling.

## Folders
- `scripts/` Python package with pipeline utilities
- `config/` layered TOML config (`studio.toml` -> `user.toml`)
- `logs/`, `cache/`, `exports/`, `temp/` standard app dirs
- `modules/` Maya `.mod` file to add to `MAYA_MODULE_PATH`
- `tests/` pytest

## Dev
1. Activate venv (see next steps printed by setup).
2. `pip install -U pip ruff pytest toml pyyaml`
3. Open in VSCode; run "Python: Unit tests (venv)" or attach to mayapy.
"""

FILESYS_PY = """\
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import os, json, tempfile, shutil, hashlib, time
from typing import Any

@dataclass
class ProjectPaths:
    root: Path
    def __post_init__(self):
        self.config  = self.root / "config"
        self.logs    = self.root / "logs"
        self.cache   = self.root / "cache"
        self.exports = self.root / "exports"
        self.temp    = self.root / "temp"
        for p in (self.config, self.logs, self.cache, self.exports, self.temp):
            p.mkdir(parents=True, exist_ok=True)

def write_txt(path: Path, text: str, encoding="utf-8", backup=True):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    if backup and path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak.{ts}"))
    os.replace(str(tmp), str(path))

def write_bytes(path: Path, data: bytes, backup=True):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    if backup and path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak.{ts}"))
    os.replace(str(tmp), str(path))

def write_json(path: Path, data: Any, indent: int = 2, sort_keys: bool = True, ensure_ascii: bool = False, default=None, backup: bool = True, create_dirs: bool = True, final_newline: bool = True,) -> Path:
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii, default=default)
    with open(tmp, "w", encoding="utf-8", newline="\\n") as f:
        f.write(text)
        if final_newline:
            f.write("\\n")
        f.flush()
        os.fsync(f.fileno())
    if backup and path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.name}.bak.{ts}"))
    os.replace(tmp, path)

def sha256_file(path: Path, chunk=1_048_576) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def fingerprint(path: Path) -> dict:
    st = Path(path).stat()
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime), "sha256": sha256_file(path)}

@contextmanager
def temp_workspace(paths: ProjectPaths, prefix="job-"):
    with tempfile.TemporaryDirectory(prefix=prefix, dir=str(paths.temp)) as d:
        yield Path(d)

DEFAULT_CONFIG = {
    "config_version": 1,
    "export": {"format": "usd", "dir": "{APP_EXPORTS}/usd"},
    "cache": {"dir": "{APP_CACHE}"},
    "logging": {"level": "INFO"},
}

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _read_any(p: Path) -> dict:
    if not p.exists(): return {}
    suf = p.suffix.lower()
    if suf == ".toml":
        import toml; return toml.load(p.open("r", encoding="utf-8"))
    if suf in (".yaml", ".yml"):
        import yaml; return yaml.safe_load(p.open("r", encoding="utf-8")) or {}
    return json.loads(p.read_text(encoding="utf-8"))

def _maya_scene_dir() -> str:
    try:
        import maya.cmds as cmds
        p = cmds.file(q=True, sn=True)
        from pathlib import Path as _P
        return str(_P(p).parent) if p else str(_P.home())
    except Exception:
        from pathlib import Path as _P
        return str(_P.home())

def _resolve_tokens(cfg: dict, root: Path) -> dict:
    token_map = {
        "APP_ROOT": str(root),
        "APP_CONFIG": str(root / "config"),
        "APP_CACHE": str(root / "cache"),
        "APP_EXPORTS": str(root / "exports"),
        "SCENE_DIR": _maya_scene_dir(),
    }
    def ex(v):
        if isinstance(v, str):
            s = os.path.expandvars(v)
            for k, vv in token_map.items():
                s = s.replace("{" + k + "}", vv)
            return s
        if isinstance(v, dict): return {k: ex(x) for k, x in v.items()}
        if isinstance(v, list): return [ex(x) for x in v]
        return v
    return ex(cfg)

def _validate(cfg: dict):
    if "export" not in cfg or "dir" not in cfg["export"]:
        raise ValueError("Config invalid: missing export.dir")
    if cfg["export"].get("format") not in ("usd", "fbx", "abc"):
        raise ValueError("Config invalid: export.format must be usd|fbx|abc")

def load_config(root: Path) -> dict:
    cfg = {}
    _deep_update(cfg, DEFAULT_CONFIG)
    _deep_update(cfg, _read_any(root / "config" / "studio.toml"))
    _deep_update(cfg, _read_any(root / "config" / "user.toml"))
    cfg = _resolve_tokens(cfg, root)
    _validate(cfg)
    return cfg
"""

LOGGER_PY = """\
import logging, platform, getpass, os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

class ToolLogger(logging.LoggerAdapter):
    def __init__(self, name: str, logs_dir: Path, level: int = logging.DEBUG):
        '''
        level index:
            NOTSET = 0
            DEBUG = 10
            INFO = 20
            WARNING = 30
            ERROR = 40
            CRITICAL = 50
        '''
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._logfile = logs_dir / "dev_execution.log"

        base = logging.getLogger(name)
        base.setLevel(level)
        base.propagate = False

        if not base.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(scene)s | %(toolver)s | %(message)s"
            )
            fh = RotatingFileHandler(self._logfile, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
            fh.setFormatter(fmt); base.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setFormatter(fmt); base.addHandler(ch)

        super().__init__(base, self._compute_extras())

    def _compute_extras(self) -> dict:
        scene = "<unsaved>"
        try:
            import maya.cmds as cmds
            scene = cmds.file(q=True, sn=True) or "<unsaved>"
        except Exception:
            pass
        return {
            "scene": scene,
            "toolver": os.getenv("TOOL_VERSION", "dev"),
            "user": getpass.getuser(),
            "host": platform.node(),
        }

    def refresh_context(self) -> None:
        self.extra = self._compute_extras()

    def reset(self, mode: str = "rollover") -> Path:
        rfh: Optional[RotatingFileHandler] = None
        for h in self.logger.handlers:
            if isinstance(h, RotatingFileHandler):
                rfh = h
                break

        if mode == "hard":
            if rfh:
                paths = [Path(rfh.baseFilename)]
                parent = paths[0].parent
                name = paths[0].name
                paths += list(parent.glob(name + ".*"))

                for h in list(self.logger.handlers):
                    try: h.flush()
                    except Exception: pass
                    try: h.close()
                    except Exception: pass
                self.logger.handlers.clear()

                for p in paths:
                    try: p.unlink()
                    except FileNotFoundError: pass

                fmt = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(scene)s | %(toolver)s | %(message)s"
                )
                new_rfh = RotatingFileHandler(self._logfile, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
                new_rfh.setFormatter(fmt); self.logger.addHandler(new_rfh)

                ch = logging.StreamHandler(); ch.setFormatter(fmt); self.logger.addHandler(ch)
                return Path(new_rfh.baseFilename)
            return self._logfile

        if not rfh:
            raise RuntimeError("No RotatingFileHandler configured")

        rfh.acquire()
        try:
            if mode == "rollover":
                rfh.doRollover()
            else:
                rfh.flush()
                if rfh.stream is None:
                    rfh.stream = rfh._open()
                rfh.stream.seek(0)
                rfh.stream.truncate()
        finally:
            rfh.release()
        return Path(rfh.baseFilename)
"""

EXCEPTIONS_PY = """\
class RigError(Exception): pass
class SelectionError(RigError): pass
class SceneStateError(RigError): pass
class DependencyMissing(RigError): pass
"""

GUARDS_PY = """\
from functools import wraps
from .exceptions import RigError

def ui_guard(label: str, logger):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except RigError as e:
                logger.warning(f"{label}: {e}")
                _notify_user(str(e))
            except Exception:
                logger.exception(f"{label}: unhandled exception")
                _notify_user(f"{label} failed. See logs for details.")
        return wrapper
    return deco

def _notify_user(msg: str):
    try:
        import maya.OpenMaya as om
        om.MGlobal.displayError(msg)
    except Exception:
        print(f"[ERROR] {msg}")
"""

EXAMPLY_PY = """\
'''
After generate the project by MLsetup.py, put this script under the project directory: <project>/scripts
'''
from pathlib import Path
import os

from rigkit.logger import ToolLogger
from rigkit.filesys import ProjectPaths, write_txt, write_json, fingerprint, load_config
from rigkit.guards import ui_guard
from rigkit.exceptions import SelectionError

ROOT = Path(os.environ.get("ML_SCRIPT_PATH", Path(__file__).resolve().parents[1]))
PATHS = ProjectPaths(ROOT)
LOG = ToolLogger("ML", PATHS.logs)

@ui_guard('example: logger', LOG)
def example_logger():
    # Clear the tool.log
    LOG.reset("hard")

    # Write debug information into <project>/logs/tool.log
    LOG.debug("Debug", extra={"op": "export"})
    LOG.warning("Warning")
    LOG.info("Info")
    LOG.error("Error")
    LOG.critical("Critical")
    LOG.log(10, 'log')

@ui_guard('example: filesys', LOG)
def example_filesys():
    # Write txt file to <project>/logs
    write_txt(PATHS.logs / 'note.txt', 'hi')
    
    # Write json file to <project>/logs
    write_json(PATHS.logs / 'note.json', [1, 2, 3])

    # Get information and hash of content
    has = fingerprint(PATHS.logs / 'note.json')
    print(has)

    # Get config from <project>/config
    config = load_config(ROOT)
    print(config)

if __name__ == "__main__":
    example_logger()
    example_filesys()
"""

STUDIO_TOML = """\
config_version = 1
[export]
format = "usd"
dir = "{APP_EXPORTS}/usd"
[logging]
level = "INFO"
"""

USER_TOML = """\
# User overrides live here. Example:
# [export]
# format = "fbx"
"""

PYPROJECT_TOML = lambda pkg: f"""\
[project]
name = "{pkg}"
version = "0.1.0"
requires-python = ">=3.8"
description = "Rig/animation tooling"
authors = [{{name = "TD"}}]

[tool.ruff]
line-length = 120
target-version = "py38"
extend-select = ["I"]

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-q"
testpaths = ["tests"]
"""

INIT_PY = """\
from .logger import *
from .filesys import *
from .exceptions import *
from .guards import *
"""

MOD_FILE = lambda name: f"""\
+ {name} 0.1.0 ..
PYTHONPATH        +:= scripts
XBMLANGPATH       +:= icons
MAYA_PLUG_IN_PATH +:= plug-ins
"""

# ---------- Utilities ----------
def run(cmd: List[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[exec] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)

def create_directory(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[Done] Wrote {path}")

def create_venv(root: Path, py: str | None = None) -> Path:
    venv_dir = root / ".venv"
    py_exe = py or sys.executable
    run([py_exe, "-m", "venv", str(venv_dir)])
    print(f"[Done] Created venv at {venv_dir}")
    return venv_dir

def write_env(root: Path, devkit_path: str, maya_location: str | None):
    env_path = root / ".env"
    lines = [
        f"ML_SCRIPT_PATH={root.resolve()}",
        f"DEVKIT_PATH={devkit_path or ''}",
        f"MAYA_LOCATION={maya_location or ''}",
        f'MAYA_MODULE_PATH={root / "modules"}'
    ]
    write_file(env_path, "\n".join(lines))

def init_venv_activation(venv_dir: Path, project_root: Path):
    is_windows = platform.system().lower().startswith("win")
    ml_path = str(project_root.resolve())
    if is_windows:
        for name in ("activate.bat", "Activate.ps1"):
            p = venv_dir / "Scripts" / name
            if p.exists():
                print(f"[Done] Patched {p}")
    else:
        act = venv_dir / "bin" / "activate"
        if act.exists():
            print(f"[Done] Patched {act}")

def pip_install(py_exe: str, packages: list[str], index_url: Optional[str] = None) -> None:
    subprocess.run([py_exe, "-m", "ensurepip", "--upgrade"], check=False)
    base = [py_exe, "-m", "pip", "install", "-U"]
    if index_url:
        base += ["--index-url", index_url]
    subprocess.check_call(base + ["pip", "setuptools", "wheel"])
    subprocess.check_call(base + packages)

def install_toml(venv_dir: Path, index_url: Optional[str] = None) -> None:
    py = str(venv_dir / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python"))
    pip_install(py, ["toml"], index_url=index_url)

def write_vscode(root: Path, pkg: str):
    settings = root / ".vscode" / "settings.json"
    tasks    = root / ".vscode" / "tasks.json"
    write_file(settings, json.dumps(VSCODE_SETTINGS(pkg), indent=2))
    write_file(tasks, json.dumps(VSCODE_TASKS, indent=2))

def init_git(root: Path, no_git: bool):
    if no_git: return
    try:
        run(["git", "--version"], check=True)
    except Exception:
        print("[warn] git not found on PATH; skipping git init.")
        return
    run(["git", "init"], cwd=root)
    write_file(root / ".gitignore", GITIGNORE)

# ---------- Main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Scaffold a Maya tooling project.")
    p.add_argument("--name", required=True, help="Project name (folder and .mod name)")
    p.add_argument("--dest", default=".", help="Destination directory (default: cwd)")
    p.add_argument("--pkg", default="toolops", help="Default python dev files under scripts/")
    p.add_argument("--devkit", default="", help="Maya DevKit path (optional)")
    p.add_argument("--maya", default="", help="Maya install path (optional)")
    p.add_argument("--python", default=sys.executable, help="Python to spawn venv")
    p.add_argument("--no-git", action="store_true", help="Skip git init")
    p.add_argument("--force", action="store_true", help="Proceed if folder exists and non-empty")
    return p.parse_args()

def main():
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    root = dest / args.name
    pkg  = args.pkg

    if root.exists() and any(root.iterdir()) and not args.force:
        print(f"[error] Folder {root} already exists and is not empty. Use --force to proceed.")
        sys.exit(2)

    print(f"[info] Provisioning project at: {root}")

    # Dependency
    create_directory(root)

    # Venv
    venv_dir = create_venv(root, py=args.python)
    init_venv_activation(venv_dir, root)
    install_toml(venv_dir)

    # .env
    write_env(root, args.devkit, args.maya)

    # .vscode
    write_vscode(root, pkg=pkg)

    # Git
    init_git(root, args.no_git)

    # Seed package + configs
    pkg_dir = root / "scripts" / pkg
    write_file(pkg_dir / "__init__.py", INIT_PY)
    write_file(pkg_dir / "filesys.py", FILESYS_PY)
    write_file(pkg_dir / "logger.py", LOGGER_PY)
    write_file(pkg_dir / "exceptions.py", EXCEPTIONS_PY)
    write_file(pkg_dir / "guards.py", GUARDS_PY)
    write_file(root / "scripts" / "example.py", EXAMPLY_PY)

    write_file(root / "config" / "studio.toml", STUDIO_TOML)
    write_file(root / "config" / "user.toml", USER_TOML)
    write_file(root / "pyproject.toml", PYPROJECT_TOML(pkg))
    write_file(root / "modules" / f"{args.name}.mod", MOD_FILE(args.name))
    write_file(root / "README.md", README_TMPL(args.name))

    # Smoke test
    print("\n[success] Bootstrap complete.")
    is_windows = platform.system().lower().startswith("win")
    if is_windows:
        act = root / ".venv" / "Scripts" / "Activate.ps1"
        alt = root / ".venv" / "Scripts" / "activate.bat"
        print(f"  - Activate venv: {act if act.exists() else alt}")
    else:
        print(f"  - source {root / '.venv' / 'bin' / 'activate'}")
    print("  - Install dev deps: pip install -U pip ruff pytest toml pyyaml")
    print("  - Open in VSCode; run tests or attach to mayapy on port 5678")

if __name__ == "__main__":
    main()

