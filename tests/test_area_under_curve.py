"""Test to verify that the area under a curve function is behaving correctly"""
import numpy as np
import pytest
from psds_eval import PSDSEval, PSDSEvalError


def test_simple_area_under_curve():
    """Ensure that the area under a curve function produces the correct area"""
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 3])
    auc = PSDSEval._auc(x, y)
    assert auc == pytest.approx(3.0), "The area calculation was incorrect"


def test_area_under_x_not_monotonically_increasing():
    """Ensure that an error is thrown when x is an increasing series"""
    x = np.array([5, 4, 2, 1, 0.5])
    y = np.array([1, 2, 3, 4, 5])
    with pytest.raises(PSDSEvalError,
                       match="non-decreasing property not verified for x"):
        PSDSEval._auc(x, y)
    with pytest.raises(PSDSEvalError,
                       match="non-decreasing property not verified for x"):
        PSDSEval._auc(x, y)


def test_area_under_y_not_monotonically_increasing():
    """Ensure an error is raised when y is an increasing series"""
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([5, 4, 2, 1, 0.5, 0])
    with pytest.raises(PSDSEvalError,
                       match="non-decreasing property not verified for y"):
        PSDSEval._auc(x, y)


def test_simple_area_under_curve_with_max():
    """Ensure area calculation is correct when a max_x value is specified"""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1.1, 2.3, 3.5, 4.2, 5.5])
    auc = PSDSEval._auc(x, y, max_x=2)
    assert auc == pytest.approx(3.4), "The area calculation was incorrect"


def test_area_under_curve_with_mismatched_arrays():
    """Check an error is thrown when x and y are unequal in length"""
    x = np.array([0.9, 1.8, 2.7, 3.6, 4.5])
    y = np.array([1.1, 2, 3, 4, 5, 7])
    with pytest.raises(PSDSEvalError,
                       match="x and y must be of equal length 5 != 6"):
        PSDSEval._auc(x, y)


def test_area_under_curve_with_2d_arrays():
    """Ensure that an error is raised when x and y are not 1-Dimensional"""
    x = np.random.randint(0, 5, (5, 5))
    y = np.random.randint(0, 5, (2, 2))
    with pytest.raises(PSDSEvalError,
                       match="x or y are not 1-dimensional numpy.ndarray"):
        PSDSEval._auc(x, y)


def test_area_under_curve_non_numpy_arrays():
    """Ensure that an error is raised when x and y are not numpy arrays"""
    x = [0., 1., 2., 3.]
    y = [0., 2., 4., 6.]
    with pytest.raises(PSDSEvalError,
                       match="must be provided as a numpy.ndarray"):
        PSDSEval._auc(x, y)
