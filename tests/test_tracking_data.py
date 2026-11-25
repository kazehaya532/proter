"""Automated tests for the stereo traffic monitor data layer.

These tests need no cameras and no YOLO weights: they use synthetic
calibrations, synthetic trajectories and, where possible, tiny generated
videos. Run with either:

    python tests/test_tracking_data.py
    python -m pytest tests/ -v
"""

import csv
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stereo_traffic_monitor as stm  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_rig(baseline=0.5, focal_px=800.0, size=(1280, 720)):
    """Synthetic stereo rig: identical cameras, right camera displaced by baseline."""
    cx, cy = size[0] / 2.0, size[1] / 2.0
    K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]], dtype=float)
    D = np.zeros(4)
    calib = {
        "image_size": list(size),
        "rms_reprojection_error_px": 0.0,
        "left": {"camera_matrix": K.tolist(), "dist_coeffs": D.tolist()},
        "right": {"camera_matrix": K.tolist(), "dist_coeffs": D.tolist()},
        "stereo": {"R": np.eye(3).tolist(), "T": [-baseline, 0.0, 0.0]},
        "coordinate_system": stm.COORDINATE_SYSTEM,
    }
    return stm.StereoRig(calib)


class FakeSource:
    """Minimal stand-in for StereoSource (no cameras opened)."""

    def __init__(self, start_ns, fps=30.0):
        self.start_ns = int(start_ns)
        self.fps = fps
        self.size = (1280, 720)
        self.live = False


def make_det(oid=1, frame_id=1, t_ns=0, xyz=(2.0, 0.0, 15.0), vel=(0.0, 0.0, 6.0)):
    x, y, z = xyz
    vx, vy, vz = vel
    return {
        "timestamp_ns": int(t_ns), "frame_id": int(frame_id), "object_id": int(oid),
        "class": "car", "confidence": 0.9,
        "world_x_m": x, "world_y_m": y, "world_z_m": z,
        "velocity_x_ms": vx, "velocity_y_ms": vy, "velocity_z_ms": vz,
        "speed_ms": math.sqrt(vx * vx + vy * vy + vz * vz),
        "left_u": 640.0, "left_v": 320.0, "right_u": 600.0, "right_v": 320.0,
        "bbox_x": 600.0, "bbox_y": 300.0, "bbox_w": 80.0, "bbox_h": 60.0,
    }


# --------------------------------------------------------------------------- #
# Tests: triangulation geometry
# --------------------------------------------------------------------------- #
def test_triangulation_roundtrip():
    """Project a known 3D point, then recover it from (u, v, disparity)."""
    rig = make_rig(baseline=0.5, focal_px=800.0, size=(1280, 720))
    point = np.array([1.0, -0.5, 10.0])          # X right, Y down, Z forward (metres)
    u = 640.0 + 800.0 * point[0] / point[2]      # cx + f * X / Z
    v = 360.0 + 800.0 * point[1] / point[2]      # cy + f * Y / Z
    disparity = 800.0 * 0.5 / point[2]           # f * baseline / Z
    recovered = rig.triangulate(u, v, disparity)
    assert np.allclose(recovered, point, atol=1e-6), f"{recovered} != {point}"


def test_rig_exports_calibration():
    rig = make_rig()
    out = rig.export()
    assert "rectified" in out and "coordinate_system" in out
    assert abs(out["rectified"]["baseline_m"] - 0.5) < 1e-6
    assert abs(out["rectified"]["focal_px"] - 800.0) < 1e-3


# --------------------------------------------------------------------------- #
# Tests: velocity estimation
# --------------------------------------------------------------------------- #
def test_motion_estimator_known_velocity():
    """A vehicle approaching at 6 m/s must yield speed ~6 m/s after warm-up."""
    est = stm.MotionEstimator(window=8)
    fps, speed_true = 30.0, 6.0
    last_pos, last_vel = None, None
    for k in range(20):
        t_ns = int(k / fps * 1e9)
        z = 40.0 - speed_true * k / fps            # approaching along Z
        pos, vel = est.update(1, t_ns, np.array([0.0, 0.0, z]))
        last_pos, last_vel = pos, vel
    assert abs(last_vel[2] + speed_true) < 0.05, f"velocity_z {last_vel[2]}"
    assert abs(np.linalg.norm(last_vel) - speed_true) < 0.05
    assert abs(last_pos[2] - (40.0 - speed_true * 19 / fps)) < 1e-9


def test_motion_estimator_resets_after_gap():
    est = stm.MotionEstimator(window=8, max_gap_s=1.0)
    for k in range(5):
        est.update(1, int(k / 30.0 * 1e9), np.array([0.0, 0.0, 30.0 - k]))
    # object lost for > 1 s, reappears: history must have been cleared
    pos, vel = est.update(1, int(5.0 * 1e9), np.array([0.0, 0.0, 10.0]))
    assert np.allclose(vel, 0.0), "velocity should be 0 right after a tracking gap"


# --------------------------------------------------------------------------- #
# Tests: CSV recording and reading back
# --------------------------------------------------------------------------- #
def test_recorder_csv_roundtrip(tmp_path=None):
    """Write detections through Recorder, read them back, compare exactly."""
    root = Path(tmp_path) if tmp_path else Path(__file__).parent / "_tmp_recording"
    root.mkdir(parents=True, exist_ok=True)
    start_ns = 1732500000_123456789            # value large enough to expose float damage
    src = FakeSource(start_ns)
    rig = make_rig()

    rec = stm.Recorder(str(root), rig, src, "0", "1", {"model": "test"}, save_video=False)
    dets = [make_det(oid=1, frame_id=1, t_ns=start_ns),
            make_det(oid=2, frame_id=1, t_ns=start_ns, xyz=(-1.2, 0.0, 21.5))]
    rec.write(1, start_ns, None, None, dets)
    rec.write(2, start_ns + 33_333_333, None, None,
              [make_det(oid=1, frame_id=2, t_ns=start_ns + 33_333_333,
                        xyz=(2.5, 0.0, 15.0), vel=(0.0, 0.0, 6.0))])
    rec.close()

    csv_path = rec.dir + os.sep + "tracking_data.csv"
    by_frame = stm.load_tracking_csv(csv_path)
    assert set(by_frame) == {1, 2}
    assert len(by_frame[1]) == 2

    row1 = next(d for d in by_frame[1] if d["object_id"] == 1)
    assert row1["timestamp_ns"] == start_ns, "nanosecond timestamp changed during CSV roundtrip"
    assert row1["frame_id"] == 1 and row1["class"] == "car"
    assert abs(row1["world_x_m"] - 2.0) < 1e-6
    assert abs(row1["speed_ms"] - 6.0) < 1e-3

    # metadata must document units and the coordinate system
    import json
    meta = json.loads(Path(rec.dir, "recording_metadata.json").read_text())
    assert meta["units"]["position"] == "m"
    assert meta["coordinate_system"]["units"] == "metres"
    assert meta["csv_columns"] == stm.CSV_COLUMNS
    calib = json.loads(Path(rec.dir, "calibration.json").read_text())
    assert "rectified" in calib and "baseline_m" in calib["rectified"]

    if not tmp_path:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def test_csv_timestamp_precision_no_float_loss():
    """_INT_COLS values must survive write+read without losing ns precision."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = FakeSource(1_999_999_999_999_999_999)
        rec = stm.Recorder(str(root), make_rig(), src, "0", "1", {}, save_video=False)
        rec.write(1, src.start_ns, None, None, [make_det(1, 1, src.start_ns)])
        rec.close()
        rows = list(csv.DictReader(open(Path(rec.dir, "tracking_data.csv"), newline="")))
        assert rows[0]["timestamp_ns"] == str(src.start_ns)
        by_frame = stm.load_tracking_csv(str(Path(rec.dir, "tracking_data.csv")))
        assert by_frame[1][0]["timestamp_ns"] == src.start_ns


# --------------------------------------------------------------------------- #
# Tests: trail index (seek-correct playback)
# --------------------------------------------------------------------------- #
def test_track_index_trails_and_seeking():
    by_frame = {}
    for fid in range(1, 11):                       # object 7 seen in frames 1..10
        d = make_det(oid=7, frame_id=fid, t_ns=fid * 33_333_333, xyz=(0, 0, 30.0 - fid))
        d["left_u"] = 100.0 + fid
        by_frame[fid] = [d]
    index = stm.TrackIndex(by_frame)

    tr = index.trails(10, [7], "left", length=5)
    assert len(tr[7]) == 5
    assert tr[7][-1] == (110.0, 320.0)             # newest point last
    assert tr[7][0] == (106.0, 320.0)              # exactly `length` points back

    tr = index.trails(4, [7], "left", length=100)  # seeking backwards: no future points
    assert [p[0] for p in tr[7]] == [101.0, 102.0, 103.0, 104.0]

    assert index.trails(10, [999], "left", length=5) == {}


# --------------------------------------------------------------------------- #
# Tests: playback frame mapping (needs a tiny writable video)
# --------------------------------------------------------------------------- #
def test_playback_frame_mapping(tmp_path=None):
    """CSV frame_id is 1-based and must line up with Playback.dets(idx)."""
    import tempfile
    td = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    try:
        rig = make_rig()
        src = FakeSource(1732500000_000000000)
        rec = stm.Recorder(td, rig, src, "l.mp4", "r.mp4", {}, save_video=False)
        for fid in range(1, 11):
            rec.write(fid, src.start_ns + fid * 33_333_333, None, None,
                      [make_det(oid=3, frame_id=fid, t_ns=src.start_ns + fid * 33_333_333)])
        rec.close()

        # build a 10-frame video in the recording folder with the same frame count
        w = cv2_writer(rec.dir, "left_camera.mp4", (1280, 720), 30.0)
        if w is not None:
            import cv2 as _cv
            frame = np.zeros((720, 1280, 3), np.uint8)
            for _ in range(10):
                w.write(frame)
            w.release()
            pb = stm.Playback.from_folder(rec.dir)
            assert pb.n_frames == 10
            assert pb.has_data
            assert pb.dets(0)[0]["frame_id"] == 1, "idx 0 must map to frame_id 1"
            assert pb.dets(9)[0]["frame_id"] == 10
            frames = pb.get(3)
            assert frames is not None and frames[0].shape == (720, 1280, 3)
            trails = pb.trails(9, pb.dets(9), "left", 5)
            assert len(trails[3]) == 5
            pb.close()
        else:
            print("[skip] mp4 writer unavailable; CSV<->video mapping not exercised")
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def cv2_writer(folder, name, size, fps):
    import cv2
    w = cv2.VideoWriter(str(Path(folder, name)),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (size[0], size[1]))
    return w if w.isOpened() else None


# --------------------------------------------------------------------------- #
# Tests: calibration function (GUI + CLI shared path)
# --------------------------------------------------------------------------- #
def test_stereo_calibration_error_paths():
    """Empty or mismatched folders must raise RuntimeError (GUI-friendly)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a", Path(td) / "b"
        a.mkdir(); b.mkdir()
        try:
            stm.stereo_calibration(str(a), str(b))
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "empty folders must raise RuntimeError"

        import cv2
        cv2.imwrite(str(a / "x.png"), np.zeros((120, 160), np.uint8))
        try:
            stm.stereo_calibration(str(a), str(b))
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "mismatched image counts must raise RuntimeError"


def _render_board(img, K, rvec, tvec, cols, rows, square):
    """Draw a (cols+1)x(rows+1) checkerboard seen by camera (K, rvec, tvec)."""
    import cv2
    for i in range(cols + 1):
        for j in range(rows + 1):
            if (i + j) % 2 != 0:
                continue
            sq = np.array([[i, j, 0], [i + 1, j, 0],
                           [i + 1, j + 1, 0], [i, j + 1, 0]], np.float32) * square
            pts, _ = cv2.projectPoints(sq, rvec, tvec, K, None)
            cv2.fillPoly(img, [pts.reshape(-1, 2).astype(np.int32)], 0)


def test_stereo_calibration_synthetic():
    """Render paired chessboard photos, calibrate, and recover the baseline."""
    import tempfile
    import cv2
    cols, rows, square, baseline_true = 9, 6, 0.025, 0.5
    size, f = (1280, 720), 800.0
    ss = 2                                   # 2x supersampling -> accurate corners
    big = (size[0] * ss, size[1] * ss)
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], float)
    K_big = K.copy()
    K_big[0, :] *= ss
    K_big[1, :] *= ss
    T_known = np.array([-baseline_true, 0.0, 0.0])   # right cam at +X in left frame

    with tempfile.TemporaryDirectory() as td:
        dl, dr = Path(td) / "left", Path(td) / "right"
        dl.mkdir(); dr.mkdir()
        for k in range(10):
            z = 0.95 + 0.04 * k              # keep the board inside BOTH views
            t = np.array([-0.125 + 0.04 * np.sin(0.9 * k),
                          -0.0875 + 0.03 * np.cos(1.1 * k), z])
            rvec = np.array([0.08 * np.sin(0.7 * k), 0.10 * np.cos(0.5 * k),
                             0.04 * np.sin(k)])
            img_l = np.full((big[1], big[0]), 255, np.uint8)
            img_r = np.full((big[1], big[0]), 255, np.uint8)
            _render_board(img_l, K_big, rvec, t, cols, rows, square)
            _render_board(img_r, K_big, rvec, t + T_known, cols, rows, square)
            img_l = cv2.resize(img_l, size, interpolation=cv2.INTER_AREA)
            img_r = cv2.resize(img_r, size, interpolation=cv2.INTER_AREA)
            img_l = cv2.GaussianBlur(img_l, (3, 3), 0.6)   # soften edges for subpixel
            img_r = cv2.GaussianBlur(img_r, (3, 3), 0.6)
            cv2.imwrite(str(dl / f"img{k:02d}.png"), img_l)
            cv2.imwrite(str(dr / f"img{k:02d}.png"), img_r)

        calib, rms, baseline = stm.stereo_calibration(str(dl), str(dr), "9x6", square)
        assert rms < 2.0, f"RMS reprojection error too high: {rms:.2f} px"
        assert abs(baseline - baseline_true) < 0.05, \
            f"recovered baseline {baseline:.3f} m != {baseline_true} m"
        assert abs(calib["rms_reprojection_error_px"] - rms) < 1e-9
        assert calib["image_size"] == list(size)
        assert calib["coordinate_system"]["units"] == "metres"


# --------------------------------------------------------------------------- #
# Plain-python runner (works without pytest)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"[ok]   {name}")
        except Exception:
            failed.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + ("" if not failed else f"; failed: {', '.join(failed)}"))
    sys.exit(1 if failed else 0)
