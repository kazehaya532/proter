"""Camera/model-independent release regression checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import stereo_traffic_monitor as stm
import stereo_traffic_gui as gui


@pytest.mark.parametrize("dy", [-12, 0, 12])
def test_matching_known_disparity_with_vertical_offset(dy):
    left = np.random.default_rng(123).integers(0, 256, (240, 400), dtype=np.uint8)
    right = cv2.warpAffine(left, np.float32([[1, 0, -60], [0, 1, dy]]), (400, 240))
    disparity = stm.match_disparity(left, right, (100, 60, 200, 120), 128)
    assert disparity == pytest.approx(60, abs=1)


def test_textureless_matching_rejected():
    blank = np.full((240, 400), 127, dtype=np.uint8)
    assert stm.match_disparity(blank, blank, (100, 60, 200, 120), 128) is None


def test_out_of_bounds_matching_rejected():
    blank = np.zeros((240, 400), dtype=np.uint8)
    assert stm.match_disparity(blank, blank, (-50, 60, 200, 120), 128) is None


def test_bundled_defaults_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    app = tmp_path / "app"
    other = tmp_path / "elsewhere"
    app.mkdir()
    other.mkdir()
    (app / "Calibration").mkdir()
    name = "manual_calibration_experimental_multidistance.json"
    (app / "Calibration" / name).write_text("{}")
    (app / "yolov8n.pt").write_bytes(b"test path only")
    monkeypatch.chdir(other)
    monkeypatch.setattr(gui, "app_dir", lambda: str(app))
    assert Path(gui.default_calib_path()) == app / "Calibration" / name
    assert Path(gui.default_model_path()) == app / "yolov8n.pt"


def test_recording_captures_effective_settings_and_fit(tmp_path):
    calib = json.loads((ROOT / "Calibration" /
                       "manual_calibration_experimental_multidistance.json").read_text())
    rig = stm.StereoRig(calib)
    source = SimpleNamespace(start_ns=1_000_000_000, fps=30, live=True, capture_sizes=None)
    settings = {"max_disparity_px": 750, "min_depth_m": 0.1, "max_depth_m": 100}
    recorder = stm.Recorder(str(tmp_path), rig, source, "0", "1", {},
                            save_video=False, stereo_settings=settings)
    source.capture_sizes = {"left": [640, 480], "right": [640, 480]}
    recorder.write(1, 1_100_000_000, None, None, [])
    recorder.close()
    metadata = json.loads((Path(recorder.dir) / "recording_metadata.json").read_text())
    saved = json.loads((Path(recorder.dir) / "calibration.json").read_text())
    assert metadata["capture_resolution"] == source.capture_sizes
    assert metadata["video_resolution"] == [1280, 720]
    assert metadata["stereo_settings"] == settings
    assert metadata["experimental_depth_fit"] == calib["experimental_depth_fit"]
    assert saved["experimental_depth_fit"] == calib["experimental_depth_fit"]
