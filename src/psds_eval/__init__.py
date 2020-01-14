"""
psds is a library for the computation of the Polyphonic
Sound Detection Score that is used to evaluate the performance
of Sound Event Detection software.
"""
from psds_eval.psds import (PSDSEval, PSDSEvalError, plot_psd_roc)
from psds_eval.version import __version__

__all__ = ["PSDSEval", "PSDSEvalError", "plot_psd_roc"]
