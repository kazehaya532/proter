# Proter Windows beta

1. Extract the entire release ZIP to a writable local folder.
2. Keep `Proter.exe`, `_internal`, `yolov8n.pt`, and `Calibration` together.
3. Run `Proter.exe`. The packaged model works offline.
4. In Live, select your left/right camera indices. The input resolution appears
   in the status line once frames arrive. Processing uses the calibration size.
5. Select `Calibration/manual_calibration_experimental_multidistance.json`
   **only for the camera rig that supplied these measurements**:
   1 m lens-center spacing, unchanged mounting/focus/zoom, 1280x720 processing.
   It is selected automatically when no `calibration.json` override exists.
6. Keep Max disparity px at **750** for this trial (the 2 m reference needs
   682.6 px). Higher search limits can reduce FPS and increase false matches.
7. Start, record briefly, stop, then inspect the session in Analysis.

## Calibration limitations

The supplied JSON is an **experimental rig-specific depth correction**, not a
universal camera calibration. The operator reported satisfactory physical
results; this does not establish accuracy for other rigs or settings. Validate
independently at multiple distances and image positions before trusting readings.
The fit was derived from 2/4/6/8 m measurements. Do not assume accurate
extrapolation, X/Y coordinates, speed, or correction of camera rotation/distortion.
Moving cameras or changing capture mode/focus/zoom requires revalidation.
The matcher reports a representative patch disparity, not a guaranteed bumper
distance. Vehicles have multiple surfaces at different depths.

The status line says `EXPERIMENTAL DEPTH FIT` when the correction is active.
Recordings save the effective calibration, actual delivered input resolutions,
processing resolution, and stereo limits. Saved videos are rectified: use
Analysis for playback, rather than treating them as new raw camera inputs.
Original camera resolution cannot be recovered from saved processed videos.

## Troubleshooting

- Keep the complete `_internal` directory; never distribute only the EXE.
- Close all running copies before replacing a release folder.
- For startup failures, inspect `proter_error.log` beside the EXE.
- A diagnostic check (no real cameras) can be run in PowerShell:
  `Start-Process .\Proter.exe -ArgumentList '--smoke-test', 'smoke-test.json' -Wait`.
  Inspect the JSON for success; absence of a report indicates failure.
- Previous EXEs do not support this correction; selecting the JSON in an old
  version silently ignores its extra settings.

## Distribution

See `THIRD_PARTY_NOTICES.md` and `licenses/`. The included YOLO model and
Ultralytics software have AGPL-3.0 / enterprise licensing requirements. A notice
alone is not proof of compliance; review applicable terms before distribution.
