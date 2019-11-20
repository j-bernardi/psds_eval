"""Test the user facing functions in the PSDSEval module"""
import os
import numpy as np
import pytest
import pandas as pd
from psds_eval import PSDSEval

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.parametrize("x", [-1, 1.2, -5, float("-inf")])
def test_invalid_thresholds(x):
    with pytest.raises(ValueError, match="dtc_threshold"):
        PSDSEval(dtc_threshold=x)
    with pytest.raises(ValueError, match="cttc_threshold"):
        PSDSEval(cttc_threshold=x)
    with pytest.raises(ValueError, match="gtc_threshold"):
        PSDSEval(gtc_threshold=x)


@pytest.mark.parametrize("x", [1, 0, 0.0, 0.1, 0.6])
def test_valid_thresholds(x):
    assert PSDSEval(dtc_threshold=x)
    assert PSDSEval(cttc_threshold=x)
    assert PSDSEval(gtc_threshold=x)


def test_operating_point_with_no_ground_truth():
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    with pytest.raises(ValueError,
                       match="The ground truth cannot be set without data"):
        PSDSEval(metadata=metadata, ground_truth=None)


def test_operating_point_with_no_metadata():
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    with pytest.raises(ValueError,
                       match="Audio metadata is required"):
        PSDSEval(metadata=None, ground_truth=gt)


def test_set_ground_truth_with_no_ground_truth():
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(ValueError,
                       match="The ground truth cannot be set without data"):
        psds_eval.set_ground_truth(None, metadata)


def test_set_ground_truth_with_no_metadata():
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(ValueError, match="Audio metadata is required"):
        psds_eval.set_ground_truth(gt, None)


BAD_GT_DATA = [[], (0.12, 8), float("-inf"), {"gt": [7, 2]}]


@pytest.mark.parametrize("bad_data", BAD_GT_DATA)
def test_set_ground_truth_with_bad_ground_truth(bad_data):
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(ValueError,
                       match="The data must be provided in a pandas.DataFrame"):
        psds_eval.set_ground_truth(bad_data, metadata)


@pytest.mark.parametrize("bad_data", BAD_GT_DATA)
def test_set_ground_truth_with_bad_metadata(bad_data):
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(ValueError,
                       match="The data must be provided in a pandas.DataFrame"):
        psds_eval.set_ground_truth(gt, bad_data)


def test_full_psds():
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt, metadata=metadata)

    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts), \
        "Expected counts do not match"
    psds = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds.value == pytest.approx(0.9142857142857143), \
        "PSDS was calculated incorrectly"
