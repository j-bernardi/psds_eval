"""
psds is a library for the computation of the Polyphonic
Sound Detection Score that is used to evaluate the performance
of Sound Event Detection software.
"""
from psds_eval.psds import (PSDSEval, plot_psd_roc)

__all__ = ["PSDSEval", "plot_psd_roc"]
