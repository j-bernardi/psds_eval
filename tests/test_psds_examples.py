"""This test file runs full PSDS calculations using example data."""
import pytest
from os.path import (join, dirname, abspath)
import pandas as pd
import numpy as np
from psds_eval import PSDSEval


DATADIR = join(dirname(abspath(__file__)), "data")


@pytest.fixture(scope="session")
def metadata():
    """A function that provides test audio metadata to each test"""
    return pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")


def test_example_1_paper_icassp(metadata):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
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
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.9142857142857143), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_example_2_paper_icassp(metadata):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_2.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_2.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    exp_counts = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.29047619047619044), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_example_3_paper_icassp(metadata):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_3.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_3.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.6238095238095237), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_example_4(metadata):
    """Run PSDSEval on some sample data and ensure the results are correct"""
    det = pd.read_csv(join(DATADIR, "test_4.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_4.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [2, 0, 0, 1, 0],
        [0, 0, 0, 1, 3],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.5), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_det_on_file_no_gt():
    """Ensure that the psds metric is correct when there is no ground truth"""
    det = pd.DataFrame({"filename": ["test.wav"], "onset": [2.4],
                        "offset": [5.9], "event_label": ["c1"]})
    gt = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])
    metadata = pd.DataFrame({"filename": ["test.wav"], "duration": [10.0]})
    # Record the checksums of the incoming data
    meta_hash = pd.util.hash_pandas_object(metadata).values
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(class_names=['c1'], ground_truth=gt,
                         metadata=metadata)
    exp_counts = np.array([
        [0, 1],
        [0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.0), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(metadata).values == meta_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_two_operating_points_one_with_no_detections():
    """Tests a case where the dtc and gtc df's are empty for the second op"""
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    metadata = pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval(ground_truth=gt, metadata=metadata)
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    det2 = pd.read_csv(join(DATADIR, "test_4.det"), sep="\t")
    psds_eval.add_operating_point(det)
    psds_eval.add_operating_point(det2)
    assert psds_eval.psds(0.0, 0.0, 100.0).value == \
        pytest.approx(0.9142857142857143), \
        "PSDS value was calculated incorrectly"


def test_two_operating_points_second_has_filtered_out_gtc():
    """Tests a case where the gt coverage df becomes empty for the second op"""
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    metadata = pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval(1, 1, 1, ground_truth=gt, metadata=metadata)
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    det2 = pd.read_csv(join(DATADIR, "test_1a.det"), sep="\t")
    psds_eval.add_operating_point(det)
    psds_eval.add_operating_point(det2)
    assert psds_eval.psds(0.0, 0.0, 100.0).value == pytest.approx(0.0), \
        "PSDS value was calculated incorrectly"


def test_empty_det():
    """Run the PSDSEval class with tables that contain no detections"""
    gt = pd.DataFrame({"filename": ["test.wav"], "onset": [2.4],
                       "offset": [5.9], "event_label": ["c1"]})
    det = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])
    metadata = pd.DataFrame({"filename": ["test.wav"], "duration": [10.0]})
    # Record the checksums of the incoming data
    meta_hash = pd.util.hash_pandas_object(metadata).values
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(class_names=['c1'], metadata=metadata,
                         ground_truth=gt)
    exp_counts = np.array([
        [0, 0],
        [0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.0), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(metadata).values == meta_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_files_from_dcase(metadata):
    """Run PSDSEval on some example data from DCASE"""
    det = pd.read_csv(join(DATADIR, "Y23R6_ppquxs_247.000_257000.det"),
                      sep="\t")
    gt = pd.read_csv(join(DATADIR, "Y23R6_ppquxs_247.000_257000.gt"),
                     sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [1., 0., 1.],
        [1., 4., 0.],
        [0., 0., 0.]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.6089285714285714), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_full_dcase_validset():
    """Run PSDSEval on all the example data from DCASE"""
    det = pd.read_csv(join(DATADIR, "baseline_validation_AA_0.005.csv"),
                      sep="\t")
    gt = pd.read_csv(join(DATADIR, "baseline_validation_gt.csv"),
                     sep="\t")
    metadata = pd.read_csv(join(DATADIR, "baseline_validation_metadata.csv"),
                           sep="\t")
    # Record the checksums of the incoming data
    meta_hash = pd.util.hash_pandas_object(metadata).values
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [269, 9, 63, 41, 120, 13, 7, 18, 128, 2, 302],
        [5, 59, 4, 45, 29, 31, 35, 46, 86, 58, 416],
        [54, 17, 129, 19, 105, 13, 14, 16, 82, 20, 585],
        [37, 43, 8, 164, 56, 9, 63, 63, 87, 7, 1100],
        [45, 10, 79, 73, 278, 7, 24, 51, 154, 22, 1480],
        [14, 22, 11, 24, 30, 41, 51, 26, 62, 43, 386],
        [3, 20, 12, 136, 96, 35, 87, 103, 97, 27, 840],
        [8, 41, 13, 119, 93, 48, 135, 127, 185, 32, 662],
        [89, 120, 74, 493, 825, 203, 403, 187, 966, 89, 1340],
        [0, 83, 1, 12, 58, 27, 46, 46, 120, 67, 390],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    # Check that all the psds metrics match
    assert psds1.value == pytest.approx(0.0044306914546640595), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(metadata).values == meta_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)
