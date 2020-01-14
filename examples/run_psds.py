#!/usr/bin/env python3
"""
A script that calculates the PSDS using example data from DCASE 2019.
"""

import os
import numpy as np
import pandas as pd
from psds_eval import (PSDSEval, plot_psd_roc)

if __name__ == "__main__":
    dtc_threshold = 0.5
    gtc_threshold = 0.5
    cttc_threshold = 0.3
    alpha_ct = 0.0
    alpha_st = 0.0
    max_efpr = 100

    # Load metadata and ground truth tables
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ground_truth_csv = os.path.join(data_dir, "dcase2019t4_gt.csv")
    metadata_csv = os.path.join(data_dir, "dcase2019t4_meta.csv")
    gt_table = pd.read_csv(ground_truth_csv, sep="\t")
    meta_table = pd.read_csv(metadata_csv, sep="\t")

    # Instantiate PSDSEval
    psds_eval = PSDSEval(dtc_threshold, gtc_threshold, cttc_threshold,
                         ground_truth=gt_table, metadata=meta_table)

    # Add the operating points
    for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
        csv_file = os.path.join(data_dir, f"baseline_{th:.1f}.csv")
        det_t = pd.read_csv(os.path.join(csv_file), sep="\t")
        psds_eval.add_operating_point(det_t)
        print(f"\rOperating point {i+1} added", end=" ")

    # Calculate the PSD-Score
    psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
    print(f"\nPSD-Score: {psds.value:.5f}")

    # Plot the PSD-ROC
    plot_psd_roc(psds)
