"""Find and kill the processes a Chord Detector scheduled task launched.

Ending a task (`schtasks /end`, `Stop-ScheduledTask`) only kills the task's
own powershell.exe wrapper. The server chain it started via `cmd /c`
survives as orphans and keeps holding its port:

  backend:  cmd.exe /c "...\\backend\\.venv\\Scripts\\python.exe" "...\\backend\\main.py" >> logs\\backend.log
              -> venv launcher python.exe -> base python.exe ...\\backend\\main.py
  frontend: cmd.exe /c "npm.cmd" start >> logs\\frontend.log
              -> node npm-cli.js start -> cmd /c next start
              -> node ...\\frontend\\node_modules\\...\\next start
  tray:     pythonw.exe ...\\tray\\tray_app.py -> base pythonw.exe

So after ending a task we match its processes by command line and
tree-kill from the topmost match. Matching is by command line rather than
port on purpose: other projects (and other people's dev servers) use these
ports too. The repo root is part of every pattern so a second checkout's
servers are left alone.

Stdlib only: uninstall-tasks.ps1 runs this with the backend venv's python.

CLI:  python service_procs.py backend|frontend|tray|all [--root DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, NamedTuple

DEFAULT_ROOT = Path(__file__).resolve().parent.parent

SERVICES = ("backend", "frontend", "tray")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Proc(NamedTuple):
    pid: int
    ppid: int
    name: str
    cmdline: str


def _norm(s: str) -> str:
    """Lowercase, forward slashes to backslashes, collapse repeated
    backslashes (npm's shims produce paths like `.bin\\\\..\\next`)."""
    return re.sub(r"\\+", r"\\", s.replace("/", "\\")).lower()


def _is_python(name: str) -> bool:
    return name in ("python.exe", "pythonw.exe")


def matches(proc: Proc, root: str | Path, service: str) -> bool:
    """True if `proc` is part of the process chain `service`'s task launched
    from the checkout at `root`."""
    r = _norm(str(root)).rstrip("\\")
    name = proc.name.lower()
    cmd = _norm(proc.cmdline or "")

    if service == "backend":
        main_py = f"{r}\\backend\\main.py"
        if _is_python(name):
            return main_py in cmd
        if name == "cmd.exe":
            return main_py in cmd and f"{r}\\logs\\backend.log" in cmd
        return False

    if service == "frontend":
        if name == "cmd.exe":
            # The launcher's root: npm.cmd start with output to our log.
            return (
                f"{r}\\logs\\frontend.log" in cmd
                and "npm" in cmd
                and re.search(r"\bstart\b", cmd) is not None
            )
        if name == "node.exe":
            # The actual server. npm's own node (npm-cli.js start) has no repo
            # path in its command line; it's reached via the tree kill from
            # the cmd root, and exits on its own once next is gone.
            return (
                f"{r}\\frontend\\node_modules\\" in cmd
                and "\\next\\dist\\bin\\next" in cmd
                and re.search(r"\bstart\b", cmd) is not None
            )
        return False

    if service == "tray":
        return _is_python(name) and f"{r}\\tray\\tray_app.py" in cmd

    raise ValueError(f"unknown service: {service!r}")


def find_kill_roots(
    procs: Iterable[Proc],
    root: str | Path,
    services: Iterable[str],
    exclude_pids: Iterable[int] = (),
) -> list[int]:
    """PIDs to `taskkill /T` so that every matching process dies.

    A matching process whose ancestor also matches is dropped, since the
    tree kill from that ancestor already covers it.
    """
    procs = list(procs)
    services = list(services)
    exclude = set(exclude_pids)
    by_pid = {p.pid: p for p in procs}
    matched = {
        p.pid for p in procs
        if p.pid not in exclude and any(matches(p, root, s) for s in services)
    }

    def has_matched_ancestor(pid: int) -> bool:
        seen = {pid}
        cur = by_pid[pid].ppid
        # PIDs get reused on Windows, so a stale ppid can point anywhere;
        # the seen-set stops cycles.
        while cur in by_pid and cur not in seen:
            if cur in matched:
                return True
            seen.add(cur)
            cur = by_pid[cur].ppid
        return False

    return sorted(pid for pid in matched if not has_matched_ancestor(pid))


def list_processes() -> list[Proc]:
    """Snapshot of every process with its command line, via CIM."""
    script = (
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        check=True,
        creationflags=_NO_WINDOW,
    ).stdout.decode("utf-8", errors="replace")
    rows = json.loads(out or "[]")
    if isinstance(rows, dict):
        rows = [rows]
    return [
        Proc(
            pid=int(r["ProcessId"]),
            ppid=int(r["ParentProcessId"] or 0),
            name=r["Name"] or "",
            cmdline=r["CommandLine"] or "",
        )
        for r in rows
    ]


def kill_services(
    services: Iterable[str],
    root: str | Path = DEFAULT_ROOT,
    dry_run: bool = False,
    wait_seconds: float = 10.0,
) -> list[Proc]:
    """Tree-kill every process chain belonging to `services`, then wait
    (up to `wait_seconds`) until none are left. Returns the roots killed."""
    services = list(services)
    procs = list_processes()
    by_pid = {p.pid: p for p in procs}
    roots = find_kill_roots(procs, root, services, exclude_pids=[os.getpid()])
    if dry_run or not roots:
        return [by_pid[pid] for pid in roots]

    for pid in roots:
        # /T walks children by parent PID; "not found" just means an earlier
        # kill already took it.
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=_NO_WINDOW,
        )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not find_kill_roots(list_processes(), root, services, [os.getpid()]):
            break
        time.sleep(0.5)
    return [by_pid[pid] for pid in roots]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service", choices=(*SERVICES, "all"))
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="checkout the tasks run from (default: this one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be killed without killing it")
    args = ap.parse_args(argv)

    services = SERVICES if args.service == "all" else (args.service,)
    killed = kill_services(services, root=args.root, dry_run=args.dry_run)
    verb = "Would kill" if args.dry_run else "Killed"
    for p in killed:
        print(f"{verb} tree {p.pid} ({p.name}): {p.cmdline}")
    if not killed:
        print("No Chord Detector processes found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
