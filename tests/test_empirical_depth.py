"""Opt-in field fit: depth correction, not a full stereo calibration."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stereo_traffic_monitor import StereoRig


def calibration():
    calib = json.loads((ROOT / "Calibration" /
                        "manual_calibration_experimental_multidistance.json").read_text())
    calib.pop("experimental_depth_fit")
    return calib


def fitted_calibration():
    calib = calibration()
    calib["experimental_depth_fit"] = {
        "numerator_m_px": 1189.8461538461531,
        "disparity_offset_px": 92.7692307692308,
    }
    return calib


def test_default_depth_unchanged():
    rig = StereoRig(calibration())
    assert rig.triangulate(rig.cx, rig.cy, 349.2)[2] == pytest.approx(
        rig.f * rig.baseline / 349.2)


def test_fit_matches_field_model_without_changing_rectification():
    original = StereoRig(calibration())
    rig = StereoRig(fitted_calibration())
    assert np.array_equal(original.P1, rig.P1)
    assert np.array_equal(original.P2, rig.P2)
    for disparity, expected in zip(
            [682.6, 403.3, 297.7, 226.9],
            [2.01726701, 3.831653, 5.80608836, 8.87079199]):
        xyz = rig.triangulate(rig.cx + 10, rig.cy + 5, disparity)
        assert xyz[2] == pytest.approx(expected)
        assert xyz[0] == pytest.approx(10 * xyz[2] / rig.f)
        assert xyz[1] == pytest.approx(5 * xyz[2] / rig.f)


@pytest.mark.parametrize("disparity", [0, 50, 92.7692307692308, float("nan")])
def test_invalid_corrected_disparity_is_rejected(disparity):
    rig = StereoRig(fitted_calibration())
    xyz = rig.triangulate(rig.cx, rig.cy, disparity)
    assert np.isnan(xyz).all()
    assert not (0.1 <= xyz[2] <= 1000)  # existing pipeline's depth gate


def test_export_preserves_opt_in_fit():
    rig = StereoRig(fitted_calibration())
    restored = StereoRig(json.loads(json.dumps(rig.export())))
    assert restored.triangulate(300, 300, 297.7) == pytest.approx(
        rig.triangulate(300, 300, 297.7))


def test_trial_file_coefficients_come_from_supplied_measurements():
    calib = json.loads((ROOT / "Calibration" /
                       "manual_calibration_experimental_multidistance.json").read_text())
    fit = calib["experimental_depth_fit"]
    points = fit["reference_points"]
    depths = np.array([p["depth_m"] for p in points])
    disparities = np.array([p["left_px"][0] - p["right_px"][0] for p in points])
    coefficients = np.linalg.lstsq(
        np.column_stack([1 / depths, np.ones(len(depths))]), disparities, rcond=None)[0]
    assert coefficients == pytest.approx(
        [fit["numerator_m_px"], fit["disparity_offset_px"]])
    rig = StereoRig(calib)
    predicted = [rig.triangulate(rig.cx, rig.cy, d)[2] for d in disparities]
    assert predicted == pytest.approx(fit["reference_predicted_depths_m"])


@pytest.mark.parametrize("key,value", [
    ("numerator_m_px", -1), ("numerator_m_px", float("nan")),
    ("disparity_offset_px", float("inf")),
])
def test_invalid_fit_rejected(key, value):
    calib = copy.deepcopy(fitted_calibration())
    calib["experimental_depth_fit"][key] = value
    with pytest.raises(ValueError):
        StereoRig(calib)
