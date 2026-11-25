#!/usr/bin/env python3
"""Entry point for the packaged Proter app (PyInstaller build).

Launches the GUI and, if something fails at startup, writes the traceback to
proter_error.log next to the executable so problems can be reported/debugged.
"""
import os
import sys
import traceback


def main():
    try:
        from stereo_traffic_gui import main as gui_main
        gui_main()
    except Exception:
        base = (os.path.dirname(sys.executable)
                if getattr(sys, "frozen", False)
                else os.path.dirname(os.path.abspath(__file__)))
        try:
            with open(os.path.join(base, "proter_error.log"), "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
