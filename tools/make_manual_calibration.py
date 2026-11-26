#!/usr/bin/env python3
"""Build an approximate manual calibration.json for the stereo rig.

Use when chessboard calibration fails or gives a huge RMS error. Values are
entered by hand (measured), so distance/speed accuracy is NOT guaranteed.

Measured inputs:
  --focal-px        focal length in pixels (same for both cameras), or
  --hfov-deg        horizontal field of view of the video mode being used
  --baseline-m      distance between the two camera optical centers, metres
  --image-size      WxH of the camera mode used in Live (default 1280x720)
  --out             output JSON path (default manual_calibration.json)

Examples:
  python make_manual_calibration.py --hfov-deg 60 --baseline-m 0.20
  python make_manual_calibration.py --focal-px 1100 --baseline-m 0.20 --out dist/Proter/calibration.json

Rules of thumb:
  - Both cameras MUST use the same resolution/zoom/focus as in Live.
  - Cameras level and parallel (right camera to the RIGHT of the left one).
  - dist_coeffs are set to zero (no distortion correction is applied).
"""
import argparse
import json
import math

CX = 639.5   # default principal point for 1280 px width:  (W-1)/2
CY = 359.5   # default principal point for 720 px height: (H-1)/2


def focal_from_hfov(hfov_deg, width):
    """Pinhole focal length in pixels from horizontal FOV."""
    return width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-size", default="1280x720", help="WxH, default 1280x720")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="focal length in pixels (overrides --hfov-deg)")
    ap.add_argument("--hfov-deg", type=float, default=60.0,
                    help="horizontal FOV in degrees, default 60 (rough webcam)")
    ap.add_argument("--focal-left", type=float, default=None,
                    help="separate focal for the left camera (pixels)")
    ap.add_argument("--focal-right", type=float, default=None,
                    help="separate focal for the right camera (pixels)")
    ap.add_argument("--baseline-m", type=float, required=True,
                    help="distance between camera optical centers, metres")
    ap.add_argument("--out", default="manual_calibration.json")
    args = ap.parse_args()

    w, h = (int(v) for v in args.image_size.lower().split("x"))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    f_def = args.focal_px if args.focal_px else focal_from_hfov(args.hfov_deg, w)
    fl = args.focal_left if args.focal_left else f_def
    fr = args.focal_right if args.focal_right else f_def

    zero_d = [0.0, 0.0, 0.0, 0.0, 0.0]
    calib = {
        "image_size": [w, h],
        "left": {
            "camera_matrix": [[fl, 0.0, cx], [0.0, fl, cy], [0.0, 0.0, 1.0]],
            "dist_coeffs": list(zero_d),
        },
        "right": {
            "camera_matrix": [[fr, 0.0, cx], [0.0, fr, cy], [0.0, 0.0, 1.0]],
            "dist_coeffs": list(zero_d),
        },
        "stereo": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "T": [-args.baseline_m, 0.0, 0.0],
            "baseline_m": args.baseline_m,
        },
        "coordinate_system": {
            "origin": "optical centre of the left camera",
            "x_axis": "right in the left image",
            "y_axis": "down in the left image",
            "z_axis": "forward, along the left camera optical axis (depth)",
            "units": "metres",
            "pixel_space": "rectified images (see left_camera.mp4 / right_camera.mp4)",
        },
        "manual": True,
        "note": ("MANUAL approximation: hand-entered focal length, zero distortion, "
                 "parallel cameras. Validate distances before trusting speeds."),
    }
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"wrote {args.out}")
    print(f"  image {w}x{h}  focal L/R {fl:.1f}/{fr:.1f} px  baseline {args.baseline_m} m")


if __name__ == "__main__":
    main()
