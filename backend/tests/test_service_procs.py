"""Tests for tray/service_procs.py's process matching.

The fixture is a process table captured from a machine running all three
Chord Detector tasks, plus the neighbours that must survive a restart:
the tray itself, another project's server, and tools whose command lines
mention the repo path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tray"))

from service_procs import Proc, find_kill_roots, matches  # noqa: E402

ROOT = r"C:\Dev\chord-detector"
PY311 = r"C:\Users\kevin\AppData\Local\Programs\Python\Python311"

EXPLORER = Proc(2584, 1, "explorer.exe", r"C:\WINDOWS\Explorer.EXE")

# Tray task
TRAY_PS = Proc(33740, 2584, "powershell.exe",
               rf'"powershell.exe" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{ROOT}\scripts\run-tray.ps1"')
TRAY_VENV = Proc(55392, 33740, "pythonw.exe",
                 rf'"{ROOT}\backend\.venv\Scripts\pythonw.exe" "{ROOT}\tray\tray_app.py" ')
TRAY_BASE = Proc(28888, 55392, "pythonw.exe",
                 rf'"{PY311}\pythonw.exe" "{ROOT}\tray\tray_app.py" ')

# Backend task
BACK_PS = Proc(37472, 2584, "powershell.exe",
               rf'"powershell.exe" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{ROOT}\scripts\run-backend.ps1"')
BACK_CMD = Proc(14180, 37472, "cmd.exe",
                rf'"C:\WINDOWS\system32\cmd.exe" /c ""{ROOT}\backend\.venv\Scripts\python.exe" "{ROOT}\backend\main.py" >> "{ROOT}\logs\backend.log" 2>&1"')
BACK_VENV = Proc(34104, 14180, "python.exe",
                 rf'"{ROOT}\backend\.venv\Scripts\python.exe"  "{ROOT}\backend\main.py" ')
BACK_BASE = Proc(49944, 34104, "python.exe",
                 rf'"{PY311}\python.exe" "{ROOT}\backend\main.py" ')

# Frontend task
FRONT_PS = Proc(37096, 2584, "powershell.exe",
                rf'"powershell.exe" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{ROOT}\scripts\run-frontend.ps1"')
FRONT_CMD = Proc(39608, 37096, "cmd.exe",
                 rf'"C:\WINDOWS\system32\cmd.exe" /c ""C:\Program Files\nodejs\npm.cmd" start >> "{ROOT}\logs\frontend.log" 2>&1"')
FRONT_NPM = Proc(19476, 39608, "node.exe",
                 r'"C:\Program Files\nodejs\\node.exe"  "C:\Program Files\nodejs\\node_modules\npm\bin\npm-cli.js" start')
FRONT_SHIM = Proc(11936, 19476, "cmd.exe", r"C:\WINDOWS\system32\cmd.exe /d /s /c next start")
FRONT_NEXT = Proc(6252, 11936, "node.exe",
                  rf'"node"   "{ROOT}\frontend\node_modules\.bin\\..\next\dist\bin\next" start')

# Neighbours that must never be killed
MIX_NOTES = Proc(26820, 2584, "python.exe",
                 r'"C:\Dev\mix-notes\.venv\Scripts\python.exe" -m uvicorn app:app --port 8000')
CODEGRAPH = Proc(10400, 2584, "node.exe",
                 rf"C:\Users\kevin\AppData\Local\codegraph\current\node.exe codegraph.js serve --mcp --path {ROOT}")
NEXT_DEV = Proc(7000, 2584, "node.exe",
                rf'"node" "{ROOT}\frontend\node_modules\next\dist\bin\next" dev')
NEXT_BUILD = Proc(7001, 2584, "node.exe",
                  rf'"node" "{ROOT}\frontend\node_modules\next\dist\bin\next" build')
EDITOR = Proc(7002, 2584, "Code.exe", rf'"Code.exe" "{ROOT}\backend\main.py"')
OTHER_CHECKOUT = Proc(7003, 2584, "python.exe",
                      rf'"{ROOT}\.claude\worktrees\wt\backend\.venv\Scripts\python.exe" "{ROOT}\.claude\worktrees\wt\backend\main.py"')
OTHER_NEXT = Proc(7004, 2584, "node.exe",
                  r'"node" "C:\Dev\someone-else\node_modules\next\dist\bin\next" start')

TABLE = [
    EXPLORER,
    TRAY_PS, TRAY_VENV, TRAY_BASE,
    BACK_PS, BACK_CMD, BACK_VENV, BACK_BASE,
    FRONT_PS, FRONT_CMD, FRONT_NPM, FRONT_SHIM, FRONT_NEXT,
    MIX_NOTES, CODEGRAPH, NEXT_DEV, NEXT_BUILD, EDITOR, OTHER_CHECKOUT, OTHER_NEXT,
]


class TestMatches:
    @pytest.mark.parametrize("proc", [BACK_CMD, BACK_VENV, BACK_BASE])
    def test_backend_chain(self, proc):
        assert matches(proc, ROOT, "backend")

    @pytest.mark.parametrize("proc", [FRONT_CMD, FRONT_NEXT])
    def test_frontend_chain(self, proc):
        assert matches(proc, ROOT, "frontend")

    @pytest.mark.parametrize("proc", [TRAY_VENV, TRAY_BASE])
    def test_tray(self, proc):
        assert matches(proc, ROOT, "tray")

    @pytest.mark.parametrize("service", ["backend", "frontend", "tray"])
    @pytest.mark.parametrize("proc", [
        EXPLORER, MIX_NOTES, CODEGRAPH, NEXT_DEV, NEXT_BUILD, EDITOR,
        OTHER_CHECKOUT, OTHER_NEXT,
        # Task wrappers are ended by Task Scheduler itself, not by us.
        TRAY_PS, BACK_PS, FRONT_PS,
    ])
    def test_neighbours_never_match(self, proc, service):
        assert not matches(proc, ROOT, service)

    def test_services_dont_cross_match(self):
        for p in (TRAY_VENV, TRAY_BASE, FRONT_CMD, FRONT_NEXT):
            assert not matches(p, ROOT, "backend")
        for p in (TRAY_VENV, TRAY_BASE, BACK_CMD, BACK_BASE):
            assert not matches(p, ROOT, "frontend")
        for p in (BACK_BASE, FRONT_NEXT):
            assert not matches(p, ROOT, "tray")

    @pytest.mark.parametrize("root", [
        ROOT.lower(), ROOT + "\\", "C:/Dev/chord-detector", "c:/dev/chord-detector/",
    ])
    def test_root_spelling_is_normalized(self, root):
        assert matches(BACK_BASE, root, "backend")
        assert matches(FRONT_NEXT, root, "frontend")

    def test_other_root_does_not_match(self):
        assert not matches(BACK_BASE, r"C:\Dev\chord", "backend")
        assert not matches(BACK_BASE, r"C:\Dev\chord-detector-old", "backend")

    def test_unknown_service_raises(self):
        with pytest.raises(ValueError):
            matches(BACK_BASE, ROOT, "database")

    def test_empty_cmdline(self):
        # CIM reports no command line for processes we can't inspect.
        assert not matches(Proc(1, 0, "python.exe", ""), ROOT, "backend")


class TestFindKillRoots:
    def test_backend_kills_from_cmd_root(self):
        assert find_kill_roots(TABLE, ROOT, ["backend"]) == [BACK_CMD.pid]

    def test_frontend_kills_from_cmd_root_only(self):
        # FRONT_NEXT matches too, but sits under FRONT_CMD via unmatched npm
        # processes, so the tree kill from FRONT_CMD already covers it.
        assert find_kill_roots(TABLE, ROOT, ["frontend"]) == [FRONT_CMD.pid]

    def test_tray(self):
        assert find_kill_roots(TABLE, ROOT, ["tray"]) == [TRAY_VENV.pid]

    def test_all(self):
        assert find_kill_roots(TABLE, ROOT, ["backend", "frontend", "tray"]) == sorted(
            [BACK_CMD.pid, FRONT_CMD.pid, TRAY_VENV.pid]
        )

    def test_orphaned_children_become_roots(self):
        # The cmd root already died; the interpreters still hold the port.
        table = [p for p in TABLE if p.pid not in (BACK_CMD.pid, FRONT_CMD.pid,
                                                   FRONT_NPM.pid, FRONT_SHIM.pid)]
        assert find_kill_roots(table, ROOT, ["backend"]) == [BACK_VENV.pid]
        assert find_kill_roots(table, ROOT, ["frontend"]) == [FRONT_NEXT.pid]

    def test_exclude_pids(self):
        assert find_kill_roots(TABLE, ROOT, ["backend"], exclude_pids=[BACK_CMD.pid]) == [
            BACK_VENV.pid
        ]

    def test_nothing_running(self):
        assert find_kill_roots([EXPLORER, MIX_NOTES], ROOT, ["backend", "frontend"]) == []

    def test_ppid_cycle_terminates(self):
        # PID reuse can make a stale ppid point back down the tree.
        a = Proc(100, 200, "cmd.exe", "")
        b = Proc(200, 100, "python.exe", rf'python "{ROOT}\backend\main.py"')
        assert find_kill_roots([a, b], ROOT, ["backend"]) == [200]
