#!/usr/bin/env python3

from os.path import (join, dirname)
import numpy as np
import pandas as pd
from psds_eval import PSDSEval, plot_psd_roc

# task variables
dtc_threshold = 0.5
gtc_threshold = 0.5
cttc_threshold = 0.3
alpha_ct = 0.
alpha_st = 0.
max_efpr = 100

# load metadata and ground truth tables
data_dir = join(dirname(__file__), "data")
gt_table = pd.read_csv(join(data_dir, "dcase2019t4_gt.csv"), sep="\t")
meta_table = pd.read_csv(join(data_dir, "dcase2019t4_meta.csv"), sep="\t")

# instantiate PSDSEval
psds_eval = PSDSEval(dtc_threshold, gtc_threshold, cttc_threshold,
                     ground_truth=gt_table, metadata=meta_table)

# Add the system operating points previously generated
for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
    csv_file = join(data_dir, "baseline_{0:.1f}.csv".format(th))
    det_t = pd.read_csv(join(csv_file), sep="\t")
    psds_eval.add_operating_point(det_t)
    print("\rOperating point {} added".format(i+1), end=" ")

# compute the PSD-Score
psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
print("\nPSD-Score: {0:.5f}".format(psds.value))

# plot PSD-ROC
plot_psd_roc(psds)
