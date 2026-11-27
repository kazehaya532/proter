#!/usr/bin/env python3
"""
Stereo traffic monitor
======================
Detection -> Tracking -> Stereo correspondence -> Triangulation -> Data export

Output layout (one folder per recording):

    recording_001/
        left_camera.mp4          rectified left video  (pixel coords in the CSV refer to this)
        right_camera.mp4         rectified right video
        tracking_data.csv        one row per vehicle per frame
        calibration.json         intrinsics, extrinsics, rectification, coordinate system
        recording_metadata.json  start time, fps, resolution, sync info, units

World coordinate system (metres, origin at the LEFT camera's optical centre):
    X = right, Y = down, Z = forward (depth)

Sub-commands:
    gui        graphical interface (live view + recording playback/analysis)
    calibrate  chessboard stereo calibration -> calibration.json
    run        headless pipeline on two video files or two live cameras
    overlay    render IDs / speeds / trajectories from a recording to a video file

Requirements:  pip install opencv-python numpy ultralytics
               (the GUI additionally needs tkinter, which ships with most Python installs)
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from datetime import datetime, timezone

import cv2
import numpy as np

# One row in tracking_data.csv = one vehicle seen in one frame.
#   timestamp_ns  capture time, nanoseconds since the Unix epoch. Live cameras:
#                 wall clock taken when both frames were grabbed. Video files:
#                 recording start + frame_id / fps (see recording_metadata.json).
#   frame_id      1-based frame number - the key for syncing CSV rows to video.
#   object_id     persistent track id, valid only within ONE recording.
#   class         COCO name (car, truck, ...), see VEHICLE_CLASSES.
#   confidence    YOLO *detection* confidence, not a 3D-measurement quality score.
#   world_*       3D position/velocity in metres and m/s, in the coordinate system
#                 documented in COORDINATE_SYSTEM and calibration.json.
#   left_u/v      centre pixel in the RECTIFIED left image.
#   right_u/v     centre pixel in the RECTIFIED right image (right_u = left_u - disparity).
#   bbox_*        bounding box in the rectified LEFT image; the right image shows
#                 the same box shifted by the disparity (no separate detection).
CSV_COLUMNS = [
    "timestamp_ns", "frame_id", "object_id", "class", "confidence",
    "world_x_m", "world_y_m", "world_z_m",
    "velocity_x_ms", "velocity_y_ms", "velocity_z_ms", "speed_ms",
    "left_u", "left_v", "right_u", "right_v",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]
_INT_COLS = {"timestamp_ns", "frame_id", "object_id"}  # written/read as exact integers
_STR_COLS = {"class"}

# COCO class ids used by YOLO models
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

SPEED_UNITS = {"km/h": 3.6, "m/s": 1.0, "mph": 2.2369363}

COORDINATE_SYSTEM = {
    "origin": "optical centre of the left camera",
    "x_axis": "right in the left image",
    "y_axis": "down in the left image",
    "z_axis": "forward, along the left camera optical axis (depth)",
    "units": "metres",
    "pixel_space": "rectified images (see left_camera.mp4 / right_camera.mp4)",
}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def stereo_calibration(left_dir, right_dir, board="9x6", square_m=0.025, progress=None):
    """Stereo calibration from paired chessboard photos.

    Returns (calib_dict, rms_px, baseline_m). Raises RuntimeError when the
    folders are empty/mismatched/insufficient, so both the CLI and the GUI
    can show a clean message. `progress(str)` receives per-file status lines.
    """
    say = progress or (lambda m: print(m))
    cols, rows = (int(v) for v in board.split("x"))
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    left_files = sorted(glob.glob(os.path.join(left_dir, "*")))
    right_files = sorted(glob.glob(os.path.join(right_dir, "*")))
    if not left_files or len(left_files) != len(right_files):
        raise RuntimeError("Need the same, non-zero number of images in both folders.")

    obj_pts, pts_l, pts_r, size = [], [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    for fl, fr in zip(left_files, right_files):
        gl, gr = cv2.imread(fl, 0), cv2.imread(fr, 0)
        if gl is None or gr is None:
            continue
        size = (gl.shape[1], gl.shape[0])
        okl, cl = cv2.findChessboardCorners(gl, (cols, rows), flags)
        okr, cr = cv2.findChessboardCorners(gr, (cols, rows), flags)
        if okl and okr:
            cl = cv2.cornerSubPix(gl, cl, (11, 11), (-1, -1), crit)
            cr = cv2.cornerSubPix(gr, cr, (11, 11), (-1, -1), crit)
            obj_pts.append(objp); pts_l.append(cl); pts_r.append(cr)
            say(f"[ok]   {os.path.basename(fl)}")
        else:
            say(f"[skip] {os.path.basename(fl)} (board not found in both)")

    if len(obj_pts) < 8:
        raise RuntimeError(f"Only {len(obj_pts)} usable pairs; need at least ~10-15.")

    _, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, pts_l, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, pts_r, size, None, None)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        obj_pts, pts_l, pts_r, K1, D1, K2, D2, size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)

    calib = {
        "image_size": list(size),
        "rms_reprojection_error_px": float(rms),
        "left": {"camera_matrix": K1.tolist(), "dist_coeffs": D1.ravel().tolist()},
        "right": {"camera_matrix": K2.tolist(), "dist_coeffs": D2.ravel().tolist()},
        "stereo": {"R": R.tolist(), "T": T.ravel().tolist(),
                   "baseline_m": float(np.linalg.norm(T))},
        "coordinate_system": COORDINATE_SYSTEM,
    }
    return calib, float(rms), float(np.linalg.norm(T))


def calibrate(args):
    """CLI wrapper: stereo calibration from chessboard image pairs -> JSON."""
    try:
        calib, rms, baseline = stereo_calibration(
            args.left_dir, args.right_dir, args.board, args.square_m, progress=print)
    except RuntimeError as e:
        sys.exit(str(e))
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved {args.out}  (RMS error {rms:.3f} px, baseline {baseline:.3f} m)")


class StereoRig:
    """Rectification maps + triangulation for a calibrated stereo pair."""

    def __init__(self, calib: dict, alpha: float = 0.0):
        self.calib = calib
        self.size = tuple(calib["image_size"])  # (w, h)
        K1 = np.array(calib["left"]["camera_matrix"])
        D1 = np.array(calib["left"]["dist_coeffs"])
        K2 = np.array(calib["right"]["camera_matrix"])
        D2 = np.array(calib["right"]["dist_coeffs"])
        R = np.array(calib["stereo"]["R"])
        T = np.array(calib["stereo"]["T"]).reshape(3, 1)

        R1, R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, self.size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        self.map_l = cv2.initUndistortRectifyMap(K1, D1, R1, self.P1, self.size, cv2.CV_16SC2)
        self.map_r = cv2.initUndistortRectifyMap(K2, D2, R2, self.P2, self.size, cv2.CV_16SC2)

        self.f = float(self.P1[0, 0])
        self.cx, self.cy = float(self.P1[0, 2]), float(self.P1[1, 2])
        self.baseline = float(abs(self.P2[0, 3] / self.P2[0, 0]))
        # Optional field-depth fit in this rig's rectified pixel coordinates.
        # This changes depth only, not intrinsics, rectification, or matching.
        self.depth_fit = calib.get("experimental_depth_fit")
        if self.depth_fit is not None:
            numerator = float(self.depth_fit["numerator_m_px"])
            offset = float(self.depth_fit["disparity_offset_px"])
            if not np.isfinite(numerator) or numerator <= 0 or not np.isfinite(offset):
                raise ValueError("Experimental depth fit needs a positive finite numerator "
                                 "and a finite disparity offset.")
            self.depth_fit = (numerator, offset)

    def rectify(self, left, right):
        return (cv2.remap(left, *self.map_l, cv2.INTER_LINEAR),
                cv2.remap(right, *self.map_r, cv2.INTER_LINEAR))

    def triangulate(self, u, v, disparity):
        if self.depth_fit is None:
            z = self.f * self.baseline / disparity
        else:
            numerator, offset = self.depth_fit
            corrected_disparity = disparity - offset
            if not np.isfinite(corrected_disparity) or corrected_disparity <= 0:
                # The pipeline's existing depth-range gate rejects NaN.
                return np.full(3, np.nan)
            z = numerator / corrected_disparity
        return np.array([(u - self.cx) * z / self.f, (v - self.cy) * z / self.f, z])

    def export(self):
        out = dict(self.calib)
        out["rectified"] = {
            "P1": self.P1.tolist(), "P2": self.P2.tolist(), "Q": self.Q.tolist(),
            "focal_px": self.f, "principal_point_px": [self.cx, self.cy],
            "baseline_m": self.baseline,
        }
        out["coordinate_system"] = COORDINATE_SYSTEM
        return out


# --------------------------------------------------------------------------- #
# Frame source (video files or live cameras)
# --------------------------------------------------------------------------- #
class StereoSource:
    def __init__(self, left, right):
        self.live = str(left).isdigit() and str(right).isdigit()
        self.caps = [cv2.VideoCapture(int(s) if str(s).isdigit() else str(s))
                     for s in (left, right)]
        for name, s, cap in zip(("left", "right"), (left, right), self.caps):
            if not cap.isOpened():
                self.release()
                hint = "" if str(s).isdigit() else \
                    " (does the file exist? supported: mp4/avi/mkv/mov)"
                raise RuntimeError(f"Cannot open {name} source '{s}'.{hint}")
        self.fps = self.caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps < 1 or self.fps > 240:
            self.fps = 30.0
        self.size = (int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.capture_sizes = None  # populated from delivered frames, not driver claims
        self.start_ns = time.time_ns()
        self.idx = 0

    def read(self):
        """Returns (ok, left, right, timestamp_ns). Both cameras are grabbed
        back-to-back, then decoded, to minimise the time skew between them."""
        ok_l = self.caps[0].grab()
        ok_r = self.caps[1].grab()
        t_live = time.time_ns()
        if not (ok_l and ok_r):
            return False, None, None, 0
        _, fl = self.caps[0].retrieve()
        _, fr = self.caps[1].retrieve()
        if fl is None or fr is None:
            return False, None, None, 0
        self.capture_sizes = {"left": [fl.shape[1], fl.shape[0]],
                              "right": [fr.shape[1], fr.shape[0]]}
        self.idx += 1
        t_ns = t_live if self.live else self.start_ns + int(self.idx / self.fps * 1e9)
        return True, fl, fr, t_ns

    def release(self):
        for c in self.caps:
            c.release()


# --------------------------------------------------------------------------- #
# Stereo matching + motion estimation
# --------------------------------------------------------------------------- #
def match_disparity(gray_l, gray_r, bbox, max_disp, min_disp=1, min_score=0.5,
                    max_dy=32, grid=(4, 3)):
    """Robustly find the disparity of a left-image bbox.

    The old single-patch matcher slid one large template along a single row of
    the right image. That fails in two common ways: with slightly misaligned
    cameras (measured ~25 px vertical offset on the reference rig) the same row
    shows different content, and one patch can lock onto the background instead
    of the object, producing wildly wrong distances.

    Instead the bbox interior is covered with a grid of small patches, each
    matched in 2-D (horizontal AND vertical) in the right image. The object
    disparity is the median of the patch disparities after outlier rejection.
    Returns None when too few patches agree, so unreliable objects are dropped
    instead of reported with a confident-looking wrong number.
    """
    H, W = gray_l.shape
    x, y, w, h = bbox
    # use the bbox interior so a loose box cannot match surrounding background
    mx, my = 0.12 * w, 0.12 * h
    x0i, y0i, x1i, y1i = x + mx, y + my, x + w - mx, y + h - my
    if x1i - x0i < 16 or y1i - y0i < 16:
        x0i, y0i, x1i, y1i = x, y, x + w, y + h
    if x0i < 0 or y0i < 0 or x1i > W or y1i > H:
        return None

    # match at half resolution: 2-D search is ~8x cheaper, accuracy is plenty
    s = 0.5
    gl = cv2.resize(gray_l, (W // 2, H // 2))
    gr = cv2.resize(gray_r, (W // 2, H // 2))
    hH, hW = gl.shape

    ps = int(np.clip(min(x1i - x0i, y1i - y0i) / 4, 24, 64))  # full-res patch size
    ps_h = max(8, int(ps * s))
    max_dh, min_dh, dy_h = max_disp * s, min_disp * s, max_dy * s

    xs_c = np.linspace(x0i + ps / 2, x1i - ps / 2, grid[0])
    ys_c = np.linspace(y0i + ps / 2, y1i - ps / 2, grid[1])
    dxs = []
    for cy in ys_c:
        for cx in xs_c:
            px, py = int(cx * s), int(cy * s)
            ph0, pw0 = py - ps_h // 2, px - ps_h // 2
            if ph0 < 0 or pw0 < 0 or ph0 + ps_h > hH or pw0 + ps_h > hW:
                continue
            patch = gl[ph0:ph0 + ps_h, pw0:pw0 + ps_h]
            if float(patch.std()) < 2.0:
                continue  # constant/near-constant templates have meaningless NCC scores
            rx0 = max(0, pw0 - int(max_dh))
            rx1 = min(hW, pw0 + ps_h - int(min_dh))
            ry0 = max(0, ph0 - int(dy_h))
            ry1 = min(hH, ph0 + int(dy_h) + ps_h)
            if rx1 - rx0 < ps_h or ry1 - ry0 < ps_h:
                continue
            res = cv2.matchTemplate(gr[ry0:ry1, rx0:rx1], patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < min_score:
                continue
            lx = loc[0]
            row = res[loc[1]]
            if 0 < lx < res.shape[1] - 1:  # sub-pixel parabola fit
                a, b, c = row[lx - 1], row[lx], row[lx + 1]
                den = a - 2 * b + c
                if abs(den) > 1e-9:
                    lx = lx + 0.5 * (a - c) / den
            disparity = (pw0 - (rx0 + lx)) / s
            if min_disp <= disparity <= max_disp:
                dxs.append(disparity)

    if len(dxs) < 3:
        return None
    arr = np.asarray(dxs)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    tol = max(4.0, 4.0 * 1.4826 * mad)
    inl = arr[np.abs(arr - med) <= tol]
    if len(inl) < max(3, int(0.3 * grid[0] * grid[1])):
        return None
    return float(np.median(inl))


class MotionEstimator:
    """Per-object position/velocity from a sliding-window least-squares line fit.

    Differencing two noisy depth measurements 1/30 s apart amplifies the noise
    ~30x, so instead we fit a straight line to the last `window` samples. The
    fitted value at the latest time is the smoothed position and the slope is the
    velocity. Velocity is reported as 0 until 3 samples exist, and the history is
    reset if the object disappears for more than `max_gap_s`.
    """

    def __init__(self, window=8, max_gap_s=1.0):
        self.window, self.max_gap_s = window, max_gap_s
        self.hist = {}  # id -> deque[(t_seconds, position)]

    def update(self, oid, t_ns, pos):
        pos = np.asarray(pos, float)
        t = t_ns / 1e9
        h = self.hist.setdefault(oid, deque(maxlen=self.window))
        if h and t - h[-1][0] > self.max_gap_s:
            h.clear()
        h.append((t, pos))
        if len(h) < 3:
            return pos, np.zeros(3)
        ts = np.array([s[0] for s in h]) - t
        if np.ptp(ts) < 1e-6:
            return pos, np.zeros(3)
        slope, intercept = np.polyfit(ts, np.array([s[1] for s in h]), 1)
        return intercept, slope


# --------------------------------------------------------------------------- #
# Per-frame pipeline (shared by CLI and GUI)
# --------------------------------------------------------------------------- #
class StereoPipeline:
    """Rectify -> detect/track (left image) -> stereo match -> triangulate -> velocity.

    process() returns (rect_left, rect_right, detections) where each detection is
    a dict keyed by CSV_COLUMNS (numbers are plain Python / float values).

    `detector` may be any callable  image -> [(x1, y1, x2, y2, track_id, cls_id, conf)]
    (handy for tests); by default Ultralytics YOLO with its built-in tracker is used.
    """

    def __init__(self, rig, model_path="yolov8n.pt", conf=0.35, tracker="bytetrack.yaml",
                 max_disp=128, min_z=2.0, max_z=80.0, detector=None):
        self.rig, self.conf, self.tracker = rig, conf, tracker
        self.max_disp, self.min_z, self.max_z = max_disp, min_z, max_z
        self.classes = list(VEHICLE_CLASSES)
        self.motion = MotionEstimator()
        self.model = None
        if detector is None:
            from ultralytics import YOLO  # imported lazily so other tools work without it
            self.model = YOLO(model_path)
            detector = self._yolo_detect
        self.detect = detector

    def _yolo_detect(self, image):
        res = self.model.track(image, persist=True, conf=self.conf, classes=self.classes,
                               tracker=self.tracker, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        return [(*map(float, b), int(i), int(c), float(s))
                for b, i, c, s in zip(xyxy, ids, cls, cf)]

    def reset(self):
        """Forget tracker / motion state (call when starting a new source)."""
        self.motion = MotionEstimator()
        if self.model is not None:
            try:
                self.model.predictor = None
            except Exception:
                pass

    def process(self, left, right, t_ns, frame_id):
        size = self.rig.size
        if (left.shape[1], left.shape[0]) != size:
            left = cv2.resize(left, size)
        if (right.shape[1], right.shape[0]) != size:
            right = cv2.resize(right, size)
        rl, rr = self.rig.rectify(left, right)
        gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)

        dets = []
        for (x1, y1, x2, y2, oid, c, cf) in self.detect(rl):
            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            d = match_disparity(gl, gr, (bx, by, bw, bh), self.max_disp)
            if d is None:
                continue  # no reliable stereo match -> no 3D measurement
            u, v = bx + bw / 2, by + bh / 2
            pos = self.rig.triangulate(u, v, d)
            if not (self.min_z <= pos[2] <= self.max_z):
                continue
            pos, vel = self.motion.update(int(oid), t_ns, pos)
            dets.append({
                "timestamp_ns": int(t_ns), "frame_id": int(frame_id), "object_id": int(oid),
                "class": VEHICLE_CLASSES.get(int(c), str(c)), "confidence": float(cf),
                "world_x_m": float(pos[0]), "world_y_m": float(pos[1]), "world_z_m": float(pos[2]),
                "velocity_x_ms": float(vel[0]), "velocity_y_ms": float(vel[1]),
                "velocity_z_ms": float(vel[2]), "speed_ms": float(np.linalg.norm(vel)),
                "left_u": float(u), "left_v": float(v),
                "right_u": float(u - d), "right_v": float(v),
                "bbox_x": float(bx), "bbox_y": float(by),
                "bbox_w": float(bw), "bbox_h": float(bh),
            })
        return rl, rr, dets


# --------------------------------------------------------------------------- #
# Recording (CSV + rectified videos + JSON metadata)
# --------------------------------------------------------------------------- #
def next_recording_dir(root):
    os.makedirs(root, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(root, f"recording_{n:03d}")):
        n += 1
    path = os.path.join(root, f"recording_{n:03d}")
    os.makedirs(path)
    return path


def det_to_row(d):
    fmt = {"confidence": "{:.3f}", "left_u": "{:.1f}", "left_v": "{:.1f}",
           "right_u": "{:.1f}", "right_v": "{:.1f}", "bbox_x": "{:.1f}", "bbox_y": "{:.1f}",
           "bbox_w": "{:.1f}", "bbox_h": "{:.1f}"}
    row = []
    for k in CSV_COLUMNS:
        v = d[k]
        if k in _INT_COLS or k in _STR_COLS:
            row.append(v)
        else:
            row.append(fmt.get(k, "{:.3f}").format(v))
    return row


class Recorder:
    def __init__(self, root, rig, source, left_name, right_name, detector_info, save_video=True,
                 stereo_settings=None):
        self.dir = next_recording_dir(root)
        self.rig = rig
        self.source = source
        self.frames, self.last_t = 0, source.start_ns
        with open(os.path.join(self.dir, "calibration.json"), "w") as f:
            json.dump(rig.export(), f, indent=2)

        self.writers = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writers = []
            for name in ("left_camera.mp4", "right_camera.mp4"):
                w = cv2.VideoWriter(os.path.join(self.dir, name), fourcc, source.fps, rig.size)
                if not w.isOpened():
                    for opened in self.writers:
                        opened.release()
                    raise RuntimeError(
                        f"Cannot open video writer for {name} (codec 'mp4v', "
                        f"{rig.size[0]}x{rig.size[1]} @ {source.fps:.1f} fps). Check that the "
                        "output folder is writable and this OpenCV build supports mp4.")
                self.writers.append(w)

        self.meta = {
            "recording_id": os.path.basename(self.dir),
            "start_time_utc": datetime.fromtimestamp(source.start_ns / 1e9, timezone.utc).isoformat(),
            "start_timestamp_ns": source.start_ns,
            "frame_rate_fps": source.fps,
            "video_resolution": list(rig.size),
            "capture_resolution": getattr(source, "capture_sizes", None),
            "stereo_settings": stereo_settings or {},
            "experimental_depth_fit": rig.calib.get("experimental_depth_fit"),
            "cameras": {"left": str(left_name), "right": str(right_name)},
            "sources_are_live": source.live,
            "video_files": ({"left": "left_camera.mp4", "right": "right_camera.mp4"}
                            if save_video else None),
            "synchronization": {
                "method": "sequential grab() of both cameras, then retrieve()",
                "timestamp_source": "wall clock at grab" if source.live
                else "start time + frame_id / fps",
                "note": "USB cameras are not hardware-synced; expect up to ~1 frame of skew.",
            },
            "detector": detector_info,
            "csv_file": "tracking_data.csv",
            "csv_columns": CSV_COLUMNS,
            "units": {"position": "m", "velocity": "m/s", "timestamp": "ns since Unix epoch",
                      "pixels": "px, rectified image space"},
            "coordinate_system": COORDINATE_SYSTEM,
            "object_id_note": "Object IDs are persistent only within this recording.",
        }
        self._save_meta()
        try:
            self.csv_file = open(os.path.join(self.dir, "tracking_data.csv"), "w", newline="")
        except OSError:
            if self.writers:
                for w in self.writers:
                    w.release()
            raise
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(CSV_COLUMNS)

    def _save_meta(self):
        self.meta["frames_processed"] = self.frames
        self.meta["end_timestamp_ns"] = self.last_t
        with open(os.path.join(self.dir, "recording_metadata.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def write(self, frame_id, t_ns, rect_left, rect_right, dets):
        self.meta["capture_resolution"] = getattr(self.source, "capture_sizes", None)
        for d in dets:
            self.writer.writerow(det_to_row(d))
        if self.writers:
            self.writers[0].write(rect_left)
            self.writers[1].write(rect_right)
        self.frames, self.last_t = frame_id, t_ns
        if frame_id % 100 == 0:
            self.csv_file.flush()

    def close(self):
        self.csv_file.close()
        if self.writers:
            for w in self.writers:
                w.release()
        self._save_meta()


# --------------------------------------------------------------------------- #
# Reading recordings back
# --------------------------------------------------------------------------- #
def load_tracking_csv(path):
    """Returns {frame_id: [detection dict, ...]} with numeric values converted."""
    by_frame = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if k in _STR_COLS:
                    d[k] = v
                elif k in _INT_COLS:
                    try:
                        d[k] = int(v)  # exact path: ns timestamps must not round-trip via float
                    except ValueError:
                        d[k] = int(float(v))  # tolerate float-formatted integers such as "1.0"
                else:
                    d[k] = float(v)
            by_frame[d["frame_id"]].append(d)
    return dict(by_frame)


class TrackIndex:
    """Stateless trajectory lookup so trails are correct after seeking."""

    def __init__(self, by_frame):
        self.tracks = defaultdict(list)
        for fid in sorted(by_frame):
            for d in by_frame[fid]:
                self.tracks[d["object_id"]].append((fid, d))
        self.fids = {oid: [f for f, _ in lst] for oid, lst in self.tracks.items()}

    def trails(self, frame_id, object_ids, camera, length):
        out = {}
        uk, vk = f"{camera}_u", f"{camera}_v"
        for oid in object_ids:
            lst, fids = self.tracks.get(oid), self.fids.get(oid)
            if not lst:
                continue
            hi = bisect_right(fids, frame_id)
            lo = bisect_left(fids, frame_id - length + 1)
            out[oid] = [(d[uk], d[vk]) for _, d in lst[lo:hi]]
        return out


class LiveTrails:
    """Trail history for live display (keeps the last `length` points per object)."""

    def __init__(self, length=200, forget_after=30):
        self.length, self.forget_after = length, forget_after
        self.hist = defaultdict(lambda: deque(maxlen=length))
        self.last_seen = {}

    def update(self, dets, frame_id):
        for d in dets:
            self.hist[d["object_id"]].append(d)
            self.last_seen[d["object_id"]] = frame_id
        for oid in [o for o, f in self.last_seen.items() if frame_id - f > self.forget_after]:
            self.hist.pop(oid, None)
            self.last_seen.pop(oid, None)

    def trails(self, camera):
        uk, vk = f"{camera}_u", f"{camera}_v"
        return {oid: [(d[uk], d[vk]) for d in h] for oid, h in self.hist.items()}


class Playback:
    """Random-access reader for a recording: video(s) + optional tracking CSV."""

    def __init__(self, left_video, right_video=None, csv_path=None):
        self.capl = cv2.VideoCapture(left_video)
        if not self.capl.isOpened():
            raise RuntimeError(f"Cannot open video: {left_video}")
        self.capr = None
        if right_video:
            self.capr = cv2.VideoCapture(right_video)
            if not self.capr.isOpened():
                self.capl.release()
                raise RuntimeError(f"Cannot open video: {right_video}")
        self.fps = self.capl.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps < 1 or self.fps > 240:
            self.fps = 30.0
        self.size = (int(self.capl.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.capl.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        n = int(self.capl.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.capr:
            nr = int(self.capr.get(cv2.CAP_PROP_FRAME_COUNT))
            n = min(n, nr) if n > 0 and nr > 0 else max(n, nr)

        self.by_frame, self.index = {}, None
        if csv_path and os.path.exists(csv_path):
            self.by_frame = load_tracking_csv(csv_path)
            self.index = TrackIndex(self.by_frame)
        if n <= 0:
            n = max(self.by_frame) if self.by_frame else 0
        if n <= 0:
            raise RuntimeError("Could not determine the number of frames in the video.")
        self.n_frames = n
        self.csv_max_frame = max(self.by_frame) if self.by_frame else 0
        if self.csv_max_frame > self.n_frames:
            print(f"[warn] tracking CSV reaches frame_id {self.csv_max_frame} but the video has "
                  f"only {self.n_frames} frames; rows beyond the last video frame are ignored.")
        self.next_idx = 0
        self.has_right = self.capr is not None
        self.has_data = self.index is not None
        self.label = os.path.dirname(os.path.abspath(left_video))

    @classmethod
    def from_folder(cls, folder):
        j = os.path.join
        left = j(folder, "left_camera.mp4")
        if not os.path.exists(left):
            raise RuntimeError(f"No left_camera.mp4 found in {folder}")
        right, csv_p = j(folder, "right_camera.mp4"), j(folder, "tracking_data.csv")
        return cls(left, right if os.path.exists(right) else None,
                   csv_p if os.path.exists(csv_p) else None)

    def get(self, idx):
        """Frames for 0-based index -> (left, right_or_None), or None on failure."""
        idx = max(0, min(self.n_frames - 1, idx))
        if idx != self.next_idx:
            self.capl.set(cv2.CAP_PROP_POS_FRAMES, idx)
            if self.capr:
                self.capr.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, left = self.capl.read()
        if not ok:
            self.next_idx = idx
            return None
        right = None
        if self.capr:
            okr, r = self.capr.read()
            right = r if okr else None
        self.next_idx = idx + 1
        return left, right

    def dets(self, idx):
        return self.by_frame.get(idx + 1, [])  # CSV frame_id is 1-based

    def trails(self, idx, dets, camera, length):
        if self.index is None or not dets:
            return {}
        return self.index.trails(idx + 1, [d["object_id"] for d in dets], camera, length)

    def close(self):
        self.capl.release()
        if self.capr:
            self.capr.release()


# --------------------------------------------------------------------------- #
# Overlay drawing (used by the GUI, `overlay` and `run --preview`)
# --------------------------------------------------------------------------- #
class OverlayOptions:
    def __init__(self, view="both", boxes=True, labels=True, trails=True, trail_len=60,
                 speed_unit="km/h", selected_id=None):
        self.view, self.boxes, self.labels, self.trails = view, boxes, labels, trails
        self.trail_len, self.speed_unit, self.selected_id = trail_len, speed_unit, selected_id


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(oid):
    hue = (int(oid) * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 200, 255]]]),
                                              cv2.COLOR_HSV2BGR)[0, 0])


def _label(img, text, org, color, fs, th, text_color=(0, 0, 0)):
    (tw, tt), base = cv2.getTextSize(text, _FONT, fs, th)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 6))
    cv2.rectangle(img, (x, y - tt - 4), (x + tw + 6, y + base), color, -1)
    cv2.putText(img, text, (x + 3, y - 1), _FONT, fs, text_color, th, cv2.LINE_AA)
    return tt + base + 4


def draw_overlay(img, dets, camera, trails, opts):
    """Draw boxes, labels, centre points and trails on `img` (modified in place)."""
    h, w = img.shape[:2]
    th = max(1, int(round(w / 700)))
    fs = max(0.4, w / 1500)
    factor = SPEED_UNITS.get(opts.speed_unit, 3.6)

    if opts.trails and trails:
        for oid, pts in trails.items():
            pts = pts[-opts.trail_len:]
            if len(pts) < 2:
                continue
            arr = np.round(np.array(pts, np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [arr], False, color_for(oid), th + 1, cv2.LINE_AA)

    for d in dets:
        oid = d["object_id"]
        col = color_for(oid)
        sel = oid == opts.selected_id
        x, y, bw, bh = d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"]
        if camera == "right":  # box was found in the left image; shift by disparity
            x += d["right_u"] - d["left_u"]
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + bw)), int(round(y + bh)))
        if opts.boxes:
            cv2.rectangle(img, p1, p2, col, th + (2 if sel else 0), cv2.LINE_AA)
        cu, cv_ = d[f"{camera}_u"], d[f"{camera}_v"]
        cv2.circle(img, (int(round(cu)), int(round(cv_))), th + 2, col, -1, cv2.LINE_AA)
        if opts.labels:
            text = (f"#{oid} {d['class']} {d['speed_ms'] * factor:.0f} {opts.speed_unit}"
                    f" | {d['world_z_m']:.1f} m")
            ty = p1[1] - 4
            (_, tt), base = cv2.getTextSize(text, _FONT, fs, th)
            if ty - tt - 4 < 0:
                ty = p1[1] + tt + base + 6
            step = _label(img, text, (p1[0], ty), col, fs, th)
            if sel:
                extra = (f"X {d['world_x_m']:+.1f}  Y {d['world_y_m']:+.1f}  Z {d['world_z_m']:.1f} m"
                         f" | v ({d['velocity_x_ms']:+.1f}, {d['velocity_z_ms']:+.1f}) m/s")
                _label(img, extra, (p1[0], ty + step), col, fs, th)
    return img


def compose_view(left, right, dets, opts, trails_left=None, trails_right=None, hud=None):
    """Build the display image: left, right or side-by-side, with overlays + HUD."""
    panels = []
    if opts.view in ("left", "both") or right is None:
        panels.append(draw_overlay(left.copy(), dets, "left", trails_left, opts))
    if opts.view in ("right", "both") and right is not None:
        panels.append(draw_overlay(right.copy(), dets, "right", trails_right, opts))
    if len(panels) == 2:
        h = panels[0].shape[0]
        if panels[1].shape[0] != h:
            s = h / panels[1].shape[0]
            panels[1] = cv2.resize(panels[1], (int(panels[1].shape[1] * s), h))
        img = np.hstack(panels)
    else:
        img = panels[0]
    if hud:
        fs = max(0.45, img.shape[1] / 1800)
        _label(img, hud, (6, int(24 * fs / 0.5)), (30, 30, 30), fs, 1, (255, 255, 255))
    return img


# --------------------------------------------------------------------------- #
# CLI: headless run / overlay export
# --------------------------------------------------------------------------- #
def run(args):
    with open(args.calibration) as f:
        rig = StereoRig(json.load(f))
    try:
        src = StereoSource(args.left, args.right)
    except RuntimeError as e:
        sys.exit(str(e))
    if src.size != rig.size:
        print(f"[warn] video is {src.size} but calibration is {rig.size}; resizing frames.")

    pipe = StereoPipeline(rig, args.model, args.conf, args.tracker,
                          args.max_disp, args.min_z, args.max_z)
    rec = Recorder(args.out, rig, src, args.left, args.right,
                    {"model": args.model, "conf": args.conf,
                     "classes": list(VEHICLE_CLASSES.values())},
                   save_video=not args.no_video,
                   stereo_settings={"max_disparity_px": args.max_disp,
                                    "min_depth_m": args.min_z, "max_depth_m": args.max_z})
    trails = LiveTrails()
    opts = OverlayOptions()
    frame_id = 0
    print(f"Recording to {rec.dir}  (Ctrl+C or 'q' to stop)")
    try:
        while True:
            ok, fl, fr, t_ns = src.read()
            if not ok:
                break
            frame_id += 1
            rl, rr, dets = pipe.process(fl, fr, t_ns, frame_id)
            rec.write(frame_id, t_ns, rl, rr, dets)
            if args.preview:
                trails.update(dets, frame_id)
                img = compose_view(rl, rr, dets, opts, trails.trails("left"),
                                   trails.trails("right"), f"frame {frame_id}")
                cv2.imshow("stereo traffic monitor", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frame_id % 100 == 0:
                print(f"frame {frame_id}")
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        rec.close()
        cv2.destroyAllWindows()
        print(f"Done. {frame_id} frames -> {rec.dir}")


def overlay(args):
    pb = Playback.from_folder(args.recording)
    if not pb.has_data:
        sys.exit("tracking_data.csv not found in that folder.")
    opts = OverlayOptions(view=args.camera, speed_unit=args.unit)
    out_path = os.path.join(args.recording, f"overlay_{args.camera}.mp4")
    out = None
    for idx in range(pb.n_frames):
        frames = pb.get(idx)
        if frames is None:
            break
        left, right = frames
        dets = pb.dets(idx)
        img = compose_view(left, right, dets, opts,
                           pb.trails(idx, dets, "left", opts.trail_len),
                           pb.trails(idx, dets, "right", opts.trail_len))
        if out is None:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), pb.fps,
                                  (img.shape[1], img.shape[0]))
        out.write(img)
    if out:
        out.release()
    pb.close()
    print("Wrote", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gui", help="open the graphical interface")
    g.set_defaults(fn=lambda a: __import__("stereo_traffic_gui").main())

    c = sub.add_parser("calibrate", help="stereo calibration from chessboard image pairs")
    c.add_argument("--left-dir", required=True)
    c.add_argument("--right-dir", required=True)
    c.add_argument("--board", default="9x6", help="inner corners, COLSxROWS")
    c.add_argument("--square-m", type=float, default=0.025, help="square size in metres")
    c.add_argument("--out", default="calibration.json")
    c.set_defaults(fn=calibrate)

    r = sub.add_parser("run", help="run detection/tracking/stereo pipeline (headless)")
    r.add_argument("--left", required=True, help="left video file or camera index")
    r.add_argument("--right", required=True, help="right video file or camera index")
    r.add_argument("--calibration", required=True)
    r.add_argument("--out", default="traffic_monitoring")
    r.add_argument("--model", default="yolov8n.pt")
    r.add_argument("--tracker", default="bytetrack.yaml")
    r.add_argument("--conf", type=float, default=0.35)
    r.add_argument("--max-disp", type=int, default=750, help="max disparity search (px)")
    r.add_argument("--min-z", type=float, default=2.0)
    r.add_argument("--max-z", type=float, default=80.0)
    r.add_argument("--no-video", action="store_true", help="do not save rectified videos")
    r.add_argument("--preview", action="store_true", help="OpenCV preview window")
    r.set_defaults(fn=run)

    o = sub.add_parser("overlay", help="render tracking overlay of a recording to a video file")
    o.add_argument("recording", help="path to recording_XXX folder")
    o.add_argument("--camera", choices=["left", "right", "both"], default="left")
    o.add_argument("--unit", choices=list(SPEED_UNITS), default="km/h")
    o.set_defaults(fn=overlay)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
