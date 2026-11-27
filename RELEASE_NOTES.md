# v0.2.5-beta — experimental field-depth correction

## Changes

- Optional `experimental_depth_fit` calibration support, retaining normal
  triangulation behavior for calibrations without this field.
- Preserve and package
  `Calibration/manual_calibration_experimental_multidistance.json`.
- Multi-patch stereo search with vertical tolerance and median aggregation;
  reject textureless templates. This is not a full camera alignment solution.
- YOLOv8n default, bundled for offline use; max disparity default 750 px.
- Live indication of input resolution and active experimental correction.
- Recording metadata includes delivered frame sizes, stereo limits, and fit.
- Separate build directories, release checksums, offline smoke-test command,
  and available dependency license notices.

## Validation

28 automated tests pass, including calibration compatibility, supplied field
fit coefficients, known synthetic disparity/vertical shifts, recording metadata,
and packaged-path selection. The synthetic chessboard test now has more varied
poses, with its original accuracy tolerances unchanged.

The operator reported satisfactory field measurements from the Python version.
The rebuilt EXE still needs operator confirmation with real cameras before
publication. An offline smoke test does not validate physical accuracy.

## Limitations

- Supplied calibration is specific to one 1 m-baseline rig and unchanged
  capture settings; its reference measurements span 2–8 m.
- It does not recover physical intrinsics/extrinsics or correct distortion,
  arbitrary misalignment, ambiguous matches, X/Y accuracy, or velocity accuracy.
- The 2 m reference has a much larger vertical discrepancy than the others.
- Median patch depth is not a guaranteed bumper measurement.
- Old EXEs silently ignore the new experimental calibration field.
- Public binary distribution requires license review, including Ultralytics/
  model AGPL-3.0 or applicable commercial terms. Project license remains an
  owner decision; do not treat bundled notices as a compliance certification.
