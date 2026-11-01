"""System-tray helper for the Chord Detector.

Polls the backend /health endpoint and shows a colored tray icon:
  green  = backend reachable
  red    = backend down
  amber  = unknown (first few seconds, or transient)

Menu actions open the UI, open the logs folder, restart the backend
or frontend via Task Scheduler, or quit the tray itself (which does
NOT stop the services — that's intentional).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
import pystray

BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:3000"
HEALTH_POLL_SECONDS = 5
HEALTH_TIMEOUT = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

BACKEND_TASK = "ChordDetectorBackend"
FRONTEND_TASK = "ChordDetectorFrontend"

STATUS_UNKNOWN = "unknown"
STATUS_OK = "ok"
STATUS_DOWN = "down"

_COLORS = {
    STATUS_OK: (76, 175, 80, 255),
    STATUS_DOWN: (229, 57, 53, 255),
    STATUS_UNKNOWN: (255, 193, 7, 255),
}

_state = {"status": STATUS_UNKNOWN}


def _make_icon(status: str) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, size - 4, size - 4), fill=_COLORS[status])
    # A plain "C" glyph on top. Fall back silently if a TrueType font isn't
    # resolvable — the colored dot alone is enough to read the status.
    try:
        font = ImageFont.truetype("arialbd.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    text = "C"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(
        ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1] - 2),
        text,
        fill=(255, 255, 255, 255),
        font=font,
    )
    return img


def _check_health() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=HEALTH_TIMEOUT)
        if r.ok and r.json().get("ok"):
            return STATUS_OK
    except Exception:
        pass
    return STATUS_DOWN


# ---- Menu actions ---------------------------------------------------------

def _open_app(icon=None, item=None) -> None:
    webbrowser.open(FRONTEND_URL)


def _open_logs(icon=None, item=None) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    os.startfile(str(LOG_DIR))  # type: ignore[attr-defined]


def _restart_task(task_name: str) -> None:
    # schtasks /end returns non-zero when the task isn't running; that's fine.
    subprocess.run(
        ["schtasks", "/end", "/tn", task_name],
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    subprocess.run(
        ["schtasks", "/run", "/tn", task_name],
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _restart_backend(icon=None, item=None) -> None:
    threading.Thread(target=_restart_task, args=(BACKEND_TASK,), daemon=True).start()


def _restart_frontend(icon=None, item=None) -> None:
    threading.Thread(target=_restart_task, args=(FRONTEND_TASK,), daemon=True).start()


def _quit(icon: pystray.Icon, item=None) -> None:
    icon.visible = False
    icon.stop()


# ---- Status polling -------------------------------------------------------

def _poll_loop(icon: pystray.Icon) -> None:
    while True:
        new_status = _check_health()
        if new_status != _state["status"]:
            _state["status"] = new_status
            icon.icon = _make_icon(new_status)
            icon.title = f"Chord Detector — {new_status}"
        time.sleep(HEALTH_POLL_SECONDS)


# ---- Entrypoint -----------------------------------------------------------

def main() -> None:
    menu = pystray.Menu(
        pystray.MenuItem("Open Chord Detector", _open_app, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Restart backend", _restart_backend),
        pystray.MenuItem("Restart frontend", _restart_frontend),
        pystray.MenuItem("Open logs folder", _open_logs),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit tray", _quit),
    )
    icon = pystray.Icon(
        "chord-detector",
        _make_icon(STATUS_UNKNOWN),
        "Chord Detector — starting",
        menu,
    )
    threading.Thread(target=_poll_loop, args=(icon,), daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()
