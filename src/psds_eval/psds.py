from warnings import warn
import pandas as pd
import numpy as np
import hashlib
from collections import namedtuple
import matplotlib.pyplot as plt

WORLD = "injected_psds_world_label"

RatesPerClass = namedtuple("RatesPerClass", ["tp_ratio", "fp_rate", "ct_rate",
                                             "effective_fp_rate"])
PSDROC = namedtuple("PSDROC", ["yp", "xp", "mean", "std"])
PSDS = namedtuple("PSDS", ["value", "plt", "alpha_st", "alpha_ct", "max_efpr"])
Thresholds = namedtuple("Thresholds", ["gtc", "dtc", "cttc"])


class PSDSEval:
    """A class to provide PSDS evaluation

    PSDS is the Polyphonic Sound Detection Score and was presented by
    Audio Analytic Labs in:
    A Framework for the Robust Evaluation of Sound Event Detection
    C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta, S. Krstulovic
    https://arxiv.org/abs/1910.08440

    Attributes:
        operating_points: An object containing all operating point data
        ground_truth: A pd.DataFrame that contains the ground truths
        metadata: A pd.DataFrame that contains the audio metadata
        class_names (list): A list of all class names in the evaluation
        threshold: (tuple): A namedTuple that contains the, gtc, dtc, and cttc
        nseconds (int): The number of seconds in the evaluation's unit of time
    """

    secs_in_uot = {"minute": 60, "hour": 3600, "day": 24 * 3600,
                   "month": 30 * 24 * 3600, "year": 365 * 24 * 3600}
    detection_cols = ["filename", "onset", "offset", "event_label"]

    def __init__(self, dtc_threshold=0.5, gtc_threshold=0.5,
                 cttc_threshold=0.3, **kwargs):
        """Initialise the PSDS evaluation

        Args:
            dtc_threshold: Detection Tolerance Criteria (DTC) threshold
            gtc_threshold: Ground Truth Intersection Criteria (GTC) threshold
            cttc_threshold: Cross-Trigger Tolerance Criteria (CTTC) threshold
            **kwargs:
            class_names: list of output class names. If not given it will be
                inferred from the ground truth table
            duration_unit: unit of time ('minute', 'hour', 'day', 'month',
                'year') for FP/CT rates report
        Raises:
            ValueError: If any of the input values are incorrect.
        """
        if dtc_threshold < 0.0 or dtc_threshold > 1.0:
            raise ValueError("dtc_threshold must be between 0 and 1")
        if cttc_threshold < 0.0 or cttc_threshold > 1.0:
            raise ValueError("cttc_threshold must be between 0 and 1")
        if gtc_threshold < 0.0 or gtc_threshold > 1.0:
            raise ValueError("gtc_threshold must be between 0 and 1")

        duration_unit = kwargs.get("duration_unit", "hour")
        if duration_unit not in self.secs_in_uot.keys():
            raise ValueError("Invalid duration_unit specified")
        self.nseconds = self.secs_in_uot[duration_unit]

        self.class_names = []
        self._update_class_names(kwargs.get("class_names", None))
        self.threshold = Thresholds(dtc=dtc_threshold, gtc=gtc_threshold,
                                    cttc=cttc_threshold)
        self.operating_points = self._operating_points_table()
        self.ground_truth = None
        self.metadata = None
        gt_t = kwargs.get("ground_truth", None)
        meta_t = kwargs.get("metadata", None)
        if gt_t is not None or meta_t is not None:
            self.set_ground_truth(gt_t, meta_t)

    @staticmethod
    def _validate_input_table(df, columns):
        """Validates given pandas.DataFrame

        Args:
            df (pandas.DataFrame): to be validated
            columns (list): Column names that should be in the df

        Raises:
            ValueError: If there is something incorrect about the df provided
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The data must be provided in a pandas.DataFrame")
        if not sorted(columns) == sorted(df.columns):
            raise ValueError("The data columns need to match the following",
                             columns)

    def num_operating_points(self):
        """Returns the number of operating point registered"""
        return len(self.operating_points.id)

    def _update_class_names(self, new_classes: list):
        """Adds new class names to the existing set

        Updates unique class names and merges them with existing class_names
        """
        if new_classes is not None and len(new_classes) > 0:
            new_classes = set(new_classes)
            _classes = set(self.class_names)
            _classes.update(new_classes)
            self.class_names = sorted(_classes)

    def set_ground_truth(self, gt_t, meta_t):
        """Validates and updates the class with a set of Ground Truths

        The Ground Truths and Metadata are used to count true positives
        (TPs), false positives (FPs) and cross-triggers (CTs) for all
        operating points when they are later added.

        Args:
            gt_t (pandas.DataFrame): A table of ground truths
            meta_t (pandas.DataFrame): A table of audio metadata information
        Raises:
            ValueError if there is an issue with the input data
        """
        if self.ground_truth is not None or self.metadata is not None:
            raise ValueError("You cannot set the ground truth more than once "
                             "per evaluation")
        if gt_t is None and meta_t is not None:
            raise ValueError("The ground truth cannot be set without data")
        if meta_t is None and gt_t is not None:
            raise ValueError("Audio metadata is required when adding ground "
                             "truths")

        self._validate_input_table(gt_t, self.detection_cols)
        self._validate_input_table(meta_t, ["filename", "duration"])
        _ground_truth = gt_t
        _metadata = meta_t

        # remove duplicated entries (possible mistake in its generation?)
        _metadata = _metadata.drop_duplicates("filename")
        metadata_t = _metadata.sort_values(by=["filename"], axis=0)
        _ground_truth = self._update_world_detections(self.detection_cols,
                                                      _ground_truth,
                                                      metadata_t)
        ground_truth_t = _ground_truth.sort_values(by=self.detection_cols[:2],
                                                   axis=0)
        ground_truth_t.dropna(inplace=True)
        ground_truth_t["duration"] = \
            ground_truth_t.offset - ground_truth_t.onset
        ground_truth_t["id"] = ground_truth_t.index

        self._update_class_names(ground_truth_t.event_label)
        self.ground_truth = ground_truth_t
        self.metadata = metadata_t

    def _init_det_table(self, det_t):
        """Validate and prepare an input detection table

        Validates and updates the a detection table with an 'id' and
        duration column.

        Args:
            det_t (pandas.DataFrame): A system's detection table

        Returns:
            A tuple with the three validated and processed tables
        """
        self._validate_input_table(det_t, self.detection_cols)
        detection_t = det_t.sort_values(by=self.detection_cols[:2], axis=0)
        detection_t["duration"] = detection_t.offset - detection_t.onset
        detection_t["id"] = detection_t.index
        return detection_t

    @staticmethod
    def _update_world_detections(columns, ground_truth, metadata):
        """Extend the ground truth with WORLD detections

        Append to each file an artificial ground truth of length equal
        to the file duration provided in the metadata table.
        """
        world_gt = [
            {k: v for k, v in zip(columns,
                                  [metadata.loc[i, 'filename'], 0.0,
                                   metadata.loc[i, 'duration'], WORLD])
             } for i in metadata.index
        ]
        if len(world_gt):
            ground_truth = ground_truth.append(world_gt, ignore_index=True)
        return ground_truth

    def _operating_point_id(self, detection_table):
        """Used to produce a unique ID for each operating point"""
        h = hashlib.sha256(pd.util.hash_pandas_object(detection_table).values)
        uid = h.hexdigest()
        if uid in self.operating_points.id.values:
            warn("A similar operating point exists, skipping this one")
            uid = ""
        return uid

    @staticmethod
    def _ground_truth_intersections(detection_t, ground_truth_t):
        """Creates a table to represent the ground truth intersections

        Returns:
            A pandas table that contains the following columns:
                inter_duration: intersection between detection and gt (s)
                det_precision: indicates what portion of a detection
                    intersect one or more ground truths of the same class
                gt_coverage: measures what proportion of a ground truth
                    is covered by one or more detections of the same class
        """

        comb_t = pd.merge(detection_t, ground_truth_t,
                          how='outer', on='filename',
                          suffixes=("_det", "_gt"))
        # cross_t contains detections that intersect one or more ground truths
        cross_t = comb_t[(comb_t.onset_det <= comb_t.offset_gt) &
                         (comb_t.onset_gt <= comb_t.offset_det) &
                         comb_t.filename.notna()].copy(deep=True)
        # Add a flag to show that GT and Event labels are of the same class
        cross_t["same_cls"] = cross_t.event_label_det == cross_t.event_label_gt

        cross_t["inter_duration"] = \
            np.minimum(cross_t.offset_det, cross_t.offset_gt) - \
            np.maximum(cross_t.onset_det, cross_t.onset_gt)
        cross_t["det_precision"] = \
            cross_t.inter_duration / cross_t.duration_det
        cross_t["gt_coverage"] = \
            cross_t.inter_duration / cross_t.duration_gt
        return cross_t

    def _detection_and_ground_truth_criterons(self, cross_t):
        """Creates GTC and DTC detection sets

        Args:
            cross_t (pandas.DataFrame): A DataFrame containing detections and
                their timings that intersect with the class's ground truths.

        Returns:
            A tuple that contains two DataFrames. The first a table of
            true positive detections that satisfy both DTC and GTC. The
            second contains only the IDs of the detections that satisfy
            the DTC.
        """

        # Detections that intersect with the the ground truths
        gt_cross_t = cross_t[cross_t.same_cls]

        # Group the duplicate detections and sum the det_precision
        if gt_cross_t.empty:
            dtc_t = pd.DataFrame(columns=["id_det", "event_label_gt",
                                          "det_precision"])
        else:
            dtc_t = gt_cross_t.groupby(
                ["id_det", "event_label_gt"]
            ).det_precision.sum().reset_index()

        dtc_ids = dtc_t[dtc_t.det_precision >= self.threshold.dtc].id_det

        # Group the duplicate detections that exist in the DTC set and sum
        gtc_t = gt_cross_t[gt_cross_t.id_det.isin(dtc_ids)].groupby(
            ["id_gt", "event_label_det"]
        ).gt_coverage.sum().reset_index()

        # Join the two into a single true positive table
        if len(dtc_t) or len(gtc_t):
            tmp = pd.merge(gt_cross_t, dtc_t, on=["id_det", "event_label_gt"],
                           suffixes=("", "_sum")
                           ).merge(gtc_t, on=["id_gt", "event_label_det"],
                                   suffixes=("", "_sum"))
        else:
            cols = gt_cross_t.columns.to_list() + \
                   ["det_precision_sum", "gt_coverage_sum"]
            tmp = pd.DataFrame(columns=cols)

        dtc_filter = tmp.det_precision_sum >= self.threshold.dtc
        gtc_filter = tmp.gt_coverage_sum >= self.threshold.gtc
        return tmp[dtc_filter & gtc_filter], dtc_ids

    def _evaluate_detections(self, tp, ct):
        """Produces a confusion matrix and detection rates for all classes"""
        n_classes = len(self.class_names) - 1  # -1 to removes the world label
        counts = np.zeros([n_classes + 1, n_classes + 1])
        tp_ratio = np.zeros(n_classes)
        fp_rate = np.zeros(n_classes)
        ct_rate = np.zeros((n_classes, n_classes))
        class_names_set = set(self.class_names)
        t_filter = self.ground_truth.event_label == WORLD
        dataset_dur = self.ground_truth[t_filter].duration.sum()
        ct_tmp = \
            ct.groupby(["event_label_det", "event_label_gt"]).filename.count()
        gt_dur = self.ground_truth.groupby("event_label").duration.sum()

        # counts is a confusion matrix
        # i, cls: detection -- j, ocls: ground truth
        for i, cls in enumerate(sorted(class_names_set.difference([WORLD]))):
            counts[i, i] = len(tp[tp.event_label_gt == cls])
            n_cls_gt = \
                self.ground_truth.groupby("event_label").filename.count()
            if cls in n_cls_gt:
                tp_ratio[i] = counts[i, i] / n_cls_gt[cls]
            for j, ocls in enumerate(sorted(class_names_set)):
                try:
                    counts[j, i] = ct_tmp[cls, ocls]
                except KeyError:
                    pass
                if ocls == WORLD:
                    fp_rate[i] = counts[j, i] * self.nseconds / dataset_dur
                elif j != i:
                    ct_rate[j, i] = counts[j, i] * self.nseconds / gt_dur[ocls]
        # move the FP in the last column
        counts[:, -1] = counts[-1]
        counts[-1] = 0
        return counts, tp_ratio, fp_rate, ct_rate

    def _cross_trigger_criterion(self, inter_t, tp_t, dtc_ids):
        """Produce a set of detections that satisfy the CTTC

        Using the main intersection table and output from the dtc function. A
        set of False Positive Cross-Triggered detections is made and then
        filtered by the CTTC threshold.

        The CTTC set consists of detections that:
            1) are not in the True Positive table
            2) intersect with ground truth of a different class (incl. WORLD)
            3) have not satisfied the detection tolerance criteria

        Args:
            inter_t (pandas.DataFrame): The table of detections and their
                ground truth intersection calculations
            tp_t (pandas.DataFrame): A detection table containing true positive
                detections.
            dtc_ids (pandas.DataFrame): A table containing a list of the uid's
                that pass the dtc.
        """

        ct_t = inter_t[~inter_t.id_det.isin(tp_t.id_det) &
                       ~inter_t.same_cls & ~inter_t.id_det.isin(dtc_ids)]

        # Group the duplicate detections and sum
        tmp = ct_t.groupby(["id_det", "event_label_gt"]).det_precision.sum()
        if len(tmp):
            ct_t = pd.merge(ct_t, tmp.reset_index(), suffixes=("", "_sum"),
                            on=["id_det", "event_label_gt"])
        else:
            ct_t["det_precision_sum"] = 0.0

        # Ensure that all world events are also collected
        cttc = ct_t[(ct_t.det_precision_sum >= self.threshold.cttc) |
                    (ct_t.event_label_gt == WORLD)]
        return cttc

    def add_operating_point(self, detections):
        """Adds a new Operating Point (OP) into the evaluation

        An operating point is defined by a system's detection results given
        some user parameters. It is expected that a user generates detection
        data from multiple operating points and then passes all data to this
        function during a single system evaluation so that a comprehensive
        result can be provided.

        Args:
            detections (pandas.DataFrame): A table of system detections
                that has the following columns:
                "filename", "onset", "offset", "event_label".
        Raises:
            ValueError: If the PSDSEval ground_truth or metadata are unset.
        """
        if self.ground_truth is None:
            raise ValueError("Ground Truth must be provided before adding the "
                             "first operating point")
        if self.metadata is None:
            raise ValueError("Audio metadata must be provided before adding "
                             "the first operating point")

        # validate and prepare tables
        det_t = self._init_det_table(detections)
        op_id = self._operating_point_id(det_t)
        if not op_id:
            return

        inter_t = self._ground_truth_intersections(det_t, self.ground_truth)
        tp, dtc_ids = self._detection_and_ground_truth_criterons(inter_t)
        cttc = self._cross_trigger_criterion(inter_t, tp, dtc_ids)

        # For the final detection count we must drop duplicates
        cttc = cttc.drop_duplicates(["id_det", "event_label_gt"])
        tp = tp.drop_duplicates("id_gt")

        cts, tp_ratio, fp_rate, ct_rate = self._evaluate_detections(tp, cttc)
        self._add_op(opid=op_id, counts=cts, tpr=tp_ratio, fpr=fp_rate,
                     ctr=ct_rate)

    @staticmethod
    def _operating_points_table():
        """Returns and empty operating point table with the correct columns"""
        return pd.DataFrame(columns=["id", "counts", "tpr", "fpr", "ctr"])

    def _add_op(self, opid, counts, tpr, fpr, ctr):
        """Adds a new operating point into the class"""
        op = {"id": opid, "counts": counts, "tpr": tpr, "fpr": fpr, "ctr": ctr}
        self.operating_points = \
            self.operating_points.append(op, ignore_index=True)

    def _del_ops(self):
        """Deletes and resets all PSDSEval operating points"""
        del self.operating_points
        self.operating_points = self._operating_points_table()

    @staticmethod
    def perform_interp(x, xp, yp):
        """Interpolate the curve (xp, yp) over the points given in x

        This interpolation function uses numpy.interp but deals with
        duplicates in xp quietly.

        Args:
            x (numpy.ndarray): a series of points at which to
                evaluate the interpolated values
            xp (numpy.ndarray): x-values of the curve to be interpolated
            yp (numpy.ndarray): y-values of the curve to be interpolated

        Returns:
            Interpolated values stored in a numpy.ndarray
        """
        new_y = np.zeros_like(x)
        sorted_idx = np.argsort(xp)
        xp_unq, idx = np.unique(xp[sorted_idx], return_index=True)
        valid_x = x < xp_unq[-1]
        new_y[valid_x] = np.interp(x[valid_x], xp_unq, yp[sorted_idx][idx])
        # fill remaining point with last tp value
        last_value = yp[sorted_idx][idx[-1]]
        new_y[~valid_x] = last_value
        # make monotonic
        new_y = np.maximum.accumulate(new_y)
        return new_y

    @staticmethod
    def step_curve(x, xp, yp):
        """Performs a custom interpolation on the ROC described by (xp, yp)

        The interpolation is performed on the given x-coordinates (x)
        and x.size >= unique(xp).size. If more than one yp value exists
        for the same xp value, only the highest yp is retained. Also yp
        is made non-decreasing so that sub optimal operating points are
        ignored.

        Args:
            x (numpy.ndarray): a series of points at which to
                evaluate the interpolated values
            xp (numpy.ndarray): x-values of the curve to be interpolated
            yp (numpy.ndarray): y-values of the curve to be interpolated

        Returns:
            numpy.ndarray: An array of interpolated y values
        """
        roc_orig = pd.DataFrame({'x': xp, 'y': yp})
        roc_valid_only = (roc_orig.groupby('x')
                          .agg('max')
                          .reset_index()
                          .sort_values(by='x'))
        if x.size < roc_valid_only.x.size:
            raise RuntimeError("x: {}, xp: {}".format(x.size, xp.size))
        # make y monotonic (given the TP/FP counting method rocs are not
        # monotonically increasing)
        roc_valid_only.y = roc_valid_only.y.cummax()
        roc_new = pd.merge(
            pd.Series(x, name='x'),
            roc_valid_only,
            how="outer",
            on="x").fillna(method='ffill').fillna(value=0)
        return roc_new.y.values

    def _effective_fp_rate(self, alpha_ct=0.):
        """Calculates effective False Positive rate (eFPR)

        Calculates the the eFPR per class applying the given weight
        to cross-triggers.

        Args:
             alpha_ct (float): cross-trigger weight in effective
                 FP rate computation
        """
        if alpha_ct < 0 or alpha_ct > 1:
            raise ValueError("alpha_ct must be between 0 and 1")

        # add a zero-point in each arr below (using np.pad)
        tpr_arr = np.stack(self.operating_points.tpr.values, axis=1)
        tpr_arr = np.pad(tpr_arr, ((0, 0), (0, 1)), "constant",
                         constant_values=0)
        fpr_arr = np.stack(self.operating_points.fpr.values, axis=1)
        fpr_arr = np.pad(fpr_arr, ((0, 0), (0, 1)), "constant",
                         constant_values=0)
        ctr_arr = np.stack(self.operating_points.ctr.values, axis=2)
        ctr_arr = np.pad(ctr_arr, ((0, 0), (0, 0), (0, 1)), "constant",
                         constant_values=0)

        efpr = fpr_arr + alpha_ct * np.nanmean(ctr_arr, axis=1)

        return RatesPerClass(tp_ratio=tpr_arr, fp_rate=fpr_arr,
                             ct_rate=ctr_arr, effective_fp_rate=efpr)

    def psd_roc_curves(self, alpha_ct, linear_interp=False):
        """Generates PSD-ROC TPR vs FPR/eFPR/CTR

        Args:
            alpha_ct (float): The weighting placed upon cross triggered FPs
            linear_interp (bool): Enables linear interpolation.

        Returns:
            A tuple containing the following ROC curves, tpr_vs_fpr,
            tpr_vs_ctr, tpr_vs_efpr.
        """
        pcr = self._effective_fp_rate(alpha_ct)
        n_classes = len(self.class_names) - 1
        # common x-axis built as union of points across classes
        fpr_points = np.unique(np.sort(pcr.fp_rate.flatten()))
        efpr_points = np.unique(np.sort(pcr.effective_fp_rate.flatten()))
        ctr_points = np.unique(np.sort(pcr.ct_rate.flatten()))
        tpr_v_fpr = np.zeros((n_classes, fpr_points.size))
        tpr_v_efpr = np.zeros((n_classes, efpr_points.size))
        tpr_v_ctr = np.zeros((n_classes, n_classes, ctr_points.size))

        _curve = self.perform_interp if linear_interp else self.step_curve
        for c in range(n_classes):
            tpr_v_fpr[c] = _curve(fpr_points, pcr.fp_rate[c], pcr.tp_ratio[c])
            tpr_v_efpr[c] = _curve(efpr_points, pcr.effective_fp_rate[c],
                                   pcr.tp_ratio[c])
            for k in range(n_classes):
                if c == k:
                    continue
                tpr_v_ctr[c, k] = _curve(ctr_points, pcr.ct_rate[c, k],
                                         pcr.tp_ratio[c])

        tpr_vs_fpr_c = PSDROC(yp=tpr_v_fpr, xp=fpr_points,
                              mean=np.nanmean(tpr_v_fpr, axis=0),
                              std=np.nanstd(tpr_v_fpr, axis=0))
        tpr_vs_efpr_c = PSDROC(yp=tpr_v_efpr, xp=efpr_points,
                               mean=np.nanmean(tpr_v_efpr, axis=0),
                               std=np.nanstd(tpr_v_efpr, axis=0))
        tpr_v_ctr = tpr_v_ctr.reshape([-1, ctr_points.size])
        tpr_vs_ctr_c = PSDROC(yp=tpr_v_ctr, xp=ctr_points,
                              mean=np.nanmean(tpr_v_ctr, axis=0),
                              std=np.nanstd(tpr_v_ctr, axis=0))

        return tpr_vs_fpr_c, tpr_vs_ctr_c, tpr_vs_efpr_c

    @staticmethod
    def _effective_tp_ratio(tpr_efpr, alpha_st):
        """Calculates the effective true positive rate (eTPR)

        Reduces a set of class ROC curves into a single Polyphonic
        Sound Detection (PSD) ROC curve.

        Args:
            tpr_efpr (numpy.ndarray): A ROC that describes the PSD-ROC
                for all classes
            alpha_st (numpy.ndarray): A weighting applied to the
                inter-class variability

        Returns:
            PSDROC: A namedTuple that describes the PSD-ROC used for the
            calculation of PSDS.
        """
        etpr = tpr_efpr.mean - alpha_st * tpr_efpr.std
        etpr[etpr < 0] = 0.0
        return PSDROC(xp=tpr_efpr.xp, yp=etpr, std=tpr_efpr.std,
                      mean=tpr_efpr.mean)

    def _psds(self, psd_roc, alpha_st, max_efpr):
        """Calculate the PSDS from the PSD ROC curve"""
        psds_curve = self._effective_tp_ratio(psd_roc, alpha_st)
        return self._auc(psds_curve.xp, psds_curve.yp, max_efpr) / max_efpr

    def psds(self, alpha_ct=0.0, alpha_st=0.0, max_efpr=None, en_interp=False):
        """Computes PSDS metric for given system

        Args:
            alpha_ct (float): cross-trigger weight in effective FP
                rate computation
            alpha_st (float): cost of instability across classes used
                to compute effective TP ratio (eTPR)
            max_efpr (float): maximum effective FP rate at which the SED
                system is evaluated (default: 100 errors per unit of time)
            en_interp (bool): if true the psds is calculated using
                linear interpolation instead of a standard staircase
                when computing PSD ROC

        Returns:
            A (PSDS) Polyphonic Sound Event Detection Score object
        """

        tpr_fpr_curve, tpr_ctr_curve, tpr_efpr_curve = \
            self.psd_roc_curves(alpha_ct, en_interp)

        if max_efpr is None:
            max_efpr = np.max(tpr_efpr_curve.xp)

        psd_roc = self._effective_tp_ratio(tpr_efpr_curve, alpha_st)
        score = self._psds(psd_roc, alpha_st, max_efpr)
        return PSDS(value=score, plt=psd_roc, alpha_st=alpha_st,
                    alpha_ct=alpha_ct, max_efpr=max_efpr)

    @staticmethod
    def _auc(x, y, max_x=None):
        """Compute area under curve described by the given x, y points.

        To avoid an overestimate the area in case of large gaps between
        points, the area is computed as sums of rectangles rather than
        trapezoids (np.trapz).

        Both x and y must be non-decreasing 1-dimensional numpy.ndarray. The
        non-decreasing property is verified if for all i in {2, ..., x.size},
        x[i-1] <= x[i]

        Args:
            x (numpy.ndarray): 1-D array containing non-decreasing
                values for x-axis
            y (numpy.ndarray): 1-D array containing non-decreasing
                values for y-axis
            max_x (float): maximum x-coordinate for area computation

        Returns:
             A float that represents the area under curve

        Raises:
            ValueError: If there is an issue with the input data
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be provided as a numpy.ndarray")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("x or y are not 1-dimensional numpy.ndarray")
        if x.size != y.size:
            raise ValueError("x and y must be of equal length "
                             "{} != {}".format(x.size, y.size))
        if np.any(np.diff(x) < 0):
            raise RuntimeError("non-decreasing property not verified for x")
        if np.any(np.diff(y) < 0):
            raise RuntimeError("non-decreasing property not verified for y")
        _x = np.array(x)
        _y = np.array(y)

        if max_x is None:
            max_x = _x.max()
        if max_x not in _x:
            # add max_x to x and the correspondent y value
            _x = np.sort(np.concatenate([_x, [max_x]]))
            max_i = int(np.argwhere(_x == max_x))
            _y = np.concatenate([_y[:max_i], [_y[max_i-1]], _y[max_i:]])
        valid_idx = _x <= max_x
        dx = np.diff(_x[valid_idx])
        _y = np.array(_y[valid_idx])[:-1]
        if dx.size != _y.size:
            raise ValueError("{} != {}".format(dx.size, _y.size))
        return np.sum(dx * _y)


def plot_psd_roc(psd, en_std=False, filename=None, figsize=None):
    """Shows (or saves) the PSD-ROC with optional standard deviation.

    When the plot is generated the area under PSD-ROC is highlighted.
    The plot is affected by the values used to compute the metric:
    max_efpr, alpha_ST and alpha_CT

    Args:
        psd (PSDS): The psd_roc that is to be plotted
        en_std (bool): if true the the plot will show the standard
            deviation curve
        filename (str): if provided a file will be saved with this name
        figsize (tuple): The figsize to be given to matplotlib
    """

    if not isinstance(psd, PSDS):
        raise RuntimeError("The psds data needs to be given as a PSDS object")

    if figsize is None:
        figsize = (7, 7)

    plt.figure(figsize=figsize)
    plt.vlines(psd.max_efpr, ymin=0, ymax=1.0, linestyles='dashed')
    plt.step(psd.plt.xp, psd.plt.yp, 'b-', label='PSD-ROC', where="post")
    if en_std:
        plt.step(psd.plt.xp,
                 np.maximum(psd.plt.mean - psd.plt.std, 0),
                 c="b", linestyle="--", where="post")
        plt.step(psd.plt.xp, psd.plt.mean + psd.plt.std,
                 c="b", linestyle="--")
    plt.fill_between(psd.plt.xp, y1=psd.plt.yp, y2=0, label="AUC",
                     alpha=0.3, color="tab:blue", linewidth=3, step="post")
    plt.xlim([0, psd.max_efpr])
    plt.ylim([0, 1.0])
    plt.legend()
    plt.ylabel("eTPR")
    plt.xlabel("eFPR")
    plt.suptitle("PSDS: {0:.5f}".format(psd.value))
    plt.title("alpha_st: {0:.2f}, alpha_ct: {1:.2f}, "
              "max_efpr: {2}".format(psd.alpha_st, psd.alpha_ct,
                                     psd.max_efpr))
    plt.grid()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
