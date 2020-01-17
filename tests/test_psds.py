"""Test the user facing functions in the PSDSEval module"""
import os
import numpy as np
import pytest
import pandas as pd
from psds_eval import PSDSEval, PSDSEvalError

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.parametrize("x", [-1, 1.2, -5, float("-inf")])
def test_invalid_thresholds(x):
    """Ensure a PSDSEvalError is raised when thresholds are invalid"""
    with pytest.raises(PSDSEvalError, match="dtc_threshold"):
        PSDSEval(dtc_threshold=x)
    with pytest.raises(PSDSEvalError, match="cttc_threshold"):
        PSDSEval(cttc_threshold=x)
    with pytest.raises(PSDSEvalError, match="gtc_threshold"):
        PSDSEval(gtc_threshold=x)


@pytest.mark.parametrize("x", [1, 0, 0.0, 0.1, 0.6])
def test_valid_thresholds(x):
    """Test the PSDSEval with a range of valid threshold values"""
    assert PSDSEval(dtc_threshold=x)
    assert PSDSEval(cttc_threshold=x)
    assert PSDSEval(gtc_threshold=x)


def tests_num_operating_points_without_any_operating_points():
    """Ensures that the eval class has no operating points when initialised"""
    psds_eval = PSDSEval()
    assert psds_eval.num_operating_points() == 0


def test_eval_class_with_no_ground_truth():
    """Ensure that PSDSEval raises a PSDSEvalError when GT is None"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    with pytest.raises(PSDSEvalError,
                       match="The ground truth cannot be set without data"):
        PSDSEval(metadata=metadata, ground_truth=None)


def test_eval_class_with_no_metadata():
    """Ensure that PSDSEval raises a PSDSEvalError when metadata is None"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    with pytest.raises(PSDSEvalError,
                       match="Audio metadata is required"):
        PSDSEval(metadata=None, ground_truth=gt)


def test_set_ground_truth_with_no_ground_truth():
    """set_ground_truth() must raise a PSDSEvalError when GT is None"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError,
                       match="The ground truth cannot be set without data"):
        psds_eval.set_ground_truth(None, metadata)


def test_set_ground_truth_with_no_metadata():
    """set_ground_truth() must raise a PSDSEvalError with None metadata"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError, match="Audio metadata is required"):
        psds_eval.set_ground_truth(gt, None)


def test_setting_ground_truth_more_than_once():
    """Ensure that an error is raised when the ground truth is set twice"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)

    with pytest.raises(PSDSEvalError, match="You cannot set the ground truth "
                                            "more than once per evaluation"):
        psds_eval.set_ground_truth(gt_t=gt, meta_t=metadata)


BAD_GT_DATA = [[], (0.12, 8), float("-inf"), {"gt": [7, 2]}]


@pytest.mark.parametrize("bad_data", BAD_GT_DATA)
def test_set_ground_truth_with_bad_ground_truth(bad_data):
    """Setting the ground truth with invalid data must raise an error"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError, match="The data must be "
                                            "provided in a pandas.DataFrame"):
        psds_eval.set_ground_truth(bad_data, metadata)


@pytest.mark.parametrize("bad_data", BAD_GT_DATA)
def test_set_ground_truth_with_bad_metadata(bad_data):
    """Setting the ground truth with invalid metadata must raise an error"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError, match="The data must be "
                                            "provided in a pandas.DataFrame"):
        psds_eval.set_ground_truth(gt, bad_data)


def test_add_operating_point_with_no_metadata():
    """Ensure that add_operating_point raises an error when metadata is none"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    psds_eval = PSDSEval(metadata=None, ground_truth=None)
    with pytest.raises(PSDSEvalError,
                       match="Ground Truth must be provided "
                             "before adding the first operating point"):
        psds_eval.add_operating_point(det)


def test_add_operating_point_with_wrong_data_format():
    """Ensure add_operating_point raises an error when the input is not a
    pandas table"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t").to_numpy()
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    with pytest.raises(PSDSEvalError, match="The data must be provided in a "
                                            "pandas.DataFrame"):
        psds_eval.add_operating_point(det)


def test_add_operating_point_with_empty_dataframe():
    """Ensure add_operating_point raises an error when given an
    incorrect table"""
    det = pd.DataFrame()
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    with pytest.raises(PSDSEvalError,
                       match="The data columns need to match the following"):
        psds_eval.add_operating_point(det)


def test_that_add_operating_point_added_a_point():
    """Ensure add_operating_point adds an operating point correctly"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    psds_eval.add_operating_point(det)
    assert psds_eval.num_operating_points() == 1
    assert psds_eval.operating_points["id"][0] == \
        "6f504797195d2df3bae13e416b8bf96ca89ec4e4e4d031dadadd72e382640387"


def test_full_psds():
    """Run a full example of the PSDSEval and test the result"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)

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
