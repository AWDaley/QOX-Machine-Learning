"""PyInstaller entry point for the QOX desktop application.

Captures startup errors that would otherwise be invisible in a windowed exe.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import traceback


def _run() -> int:
    # PyInstaller --windowed sets these streams to None on Windows. Some
    # numerical/graphics packages still expect writable text streams.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        from qox.desktop import main

        return main()
    except Exception:
        report = Path(tempfile.gettempdir()) / "QOX-startup-error.log"
        report.write_text(traceback.format_exc(), encoding="utf-8")
        if "--smoke-test" not in sys.argv:
            try:
                from tkinter import Tk, messagebox

                root = Tk()
                root.withdraw()
                messagebox.showerror("QOX startup failed", f"Details saved to:\n{report}")
                root.destroy()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
