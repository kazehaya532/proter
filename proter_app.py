#!/usr/bin/env python3
"""Entry point for the packaged Proter app (PyInstaller build).

Launches the GUI and, if something fails at startup, writes the traceback to
proter_error.log next to the executable so problems can be reported/debugged.
"""
import os
import sys
import traceback


def smoke_test(report_path):
    """Explicit offline self-check for release builds; never opens real cameras."""
    import json
    from pathlib import Path
    import tkinter as tk
    import numpy as np
    from ultralytics import YOLO
    import stereo_traffic_monitor as stm
    from stereo_traffic_gui import app_dir, default_calib_path, default_model_path

    calib_path, model_path = default_calib_path(), default_model_path()
    rig = stm.StereoRig(json.loads(Path(calib_path).read_text()))
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Offline model missing: {model_path}")
    YOLO(model_path).predict(np.zeros((320, 320, 3), dtype=np.uint8),
                             device="cpu", verbose=False)
    root = tk.Tk()
    root.withdraw()
    root.update()
    root.destroy()
    Path(report_path).write_text(json.dumps({
        "passed": True, "app_dir": app_dir(), "model": model_path,
        "calibration": calib_path, "experimental_depth_fit": rig.depth_fit is not None,
        "reference_depths_m": [
            float(rig.triangulate(rig.cx, rig.cy, d)[2])
            for d in (682.6, 403.3, 297.7, 226.9)],
        "checks": ["offline model CPU inference", "Tk startup", "calibration load"],
        "not_checked": ["physical cameras", "field accuracy", "interactive GUI operation"],
    }, indent=2), encoding="utf-8")


def main():
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
            smoke_test(sys.argv[2])
            return
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
