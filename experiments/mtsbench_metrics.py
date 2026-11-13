"""Code to compute the metrics used in the benchmark. From https://github.com/PLAN-Lab/mTSBench/tree/main/Detectors/evaluation"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import numpy as np
from prts import ts_precision, ts_recall
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d


class Metric(ABC):
    """Base class for metric implementations that score anomaly scorings against ground truth binary labels. Every
    subclass must implement :func:`~timeeval.metrics.Metric.name`, :func:`~timeeval.metrics.Metric.score`, and
    :func:`~timeeval.metrics.Metric.supports_continuous_scorings`.

    Examples
    --------
    You can implement a new TimeEval metric easily by inheriting from this base class. A simple metric, for example,
    uses a fixed threshold to get binary labels and computes the false positive rate:

    >>> from timeeval.metrics import Metric
    >>> class FPR(Metric):
    >>>     def __init__(self, threshold: float = 0.8):
    >>>         self._threshold = threshold
    >>>     @property
    >>>     def name(self) -> str:
    >>>         return f"FPR@{self._threshold}"
    >>>     def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
    >>>         y_pred = y_score >= self._threshold
    >>>         fp = np.sum(y_pred & ~y_true)
    >>>         return fp / (fp + np.sum(y_true))
    >>>     def supports_continuous_scorings(self) -> bool:
    >>>         return True

    This metric can then be used in TimeEval:

    >>> from timeeval import TimeEval
    >>> from timeeval.metrics import DefaultMetrics
    >>> timeeval = TimeEval(dmgr=..., datasets=[], algorithms=[],
    >>>                     metrics=[FPR(threshold=0.8), DefaultMetrics.ROC_AUC])
    """

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:  # type: ignore[no-untyped-def]
        y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)
        if np.unique(y_score).shape[0] == 1:
            warnings.warn("Cannot compute metric for a constant value in y_score, returning 0.0!")
            return 0.0
        return self.score(y_true, y_score)

    def _validate_scores(
        self, y_true: np.ndarray, y_score: np.ndarray, inf_is_1: bool = True, neginf_is_0: bool = True, nan_is_0: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.array(y_true).copy()
        y_score = np.array(y_score).copy()
        # check labels
        if self.supports_continuous_scorings() and y_true.dtype == np.float_ and y_score.dtype == np.int_:
            warnings.warn(
                "Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
                "y_true should be an integer array and y_score a float array!"
            )
            return self._validate_scores(y_score, y_true)

        y_true: np.ndarray = column_or_1d(y_true)  # type: ignore
        assert_all_finite(y_true)

        # check scores
        y_score: np.ndarray = column_or_1d(y_score)  # type: ignore

        check_consistent_length([y_true, y_score])
        if not self.supports_continuous_scorings():
            if y_score.dtype not in [np.int_, np.bool_]:
                raise ValueError(
                    "When using Metrics other than AUC-metric that need discrete (0 or 1) scores (like "
                    "Precision, Recall or F1-Score), the scores must be integers and should only contain "
                    "the values {0, 1}. Please consider applying a threshold to the scores!"
                )
        else:
            if not np.issubdtype(y_score.dtype, np.floating):
                raise ValueError(f"When using continuous scoring metrics, the scores must be floats!. " f"Got {y_score.dtype} instead.")

        # substitute NaNs and Infs
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        penalize_mask = np.full_like(y_score, dtype=bool, fill_value=False)
        if inf_is_1:
            y_score[inf_mask] = 1
        else:
            penalize_mask = penalize_mask | inf_mask
        if neginf_is_0:
            y_score[neginf_mask] = 0
        else:
            penalize_mask = penalize_mask | neginf_mask
        if nan_is_0:
            y_score[nan_mask] = 0
        else:
            penalize_mask = penalize_mask | nan_mask
        y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

        assert_all_finite(y_score)
        return y_true, y_score

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this metric."""
        ...

    @abstractmethod
    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Implementation of the metric's scoring function.

        Please use :func:`~timeeval.metrics.Metric.__call__` instead of calling this function directly!

        Examples
        --------

        Instantiate a metric and call it using the ``__call__`` method:

        >>> import numpy as np
        >>> from timeeval.metrics import RocAUC
        >>> metric = RocAUC(plot=False)
        >>> metric(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.5

        """
        ...

    @abstractmethod
    def supports_continuous_scorings(self) -> bool:
        """Whether this metric accepts continuous anomaly scorings as input (``True``) or binary classification
        labels (``False``)."""
        ...


class AucMetric(Metric, ABC):
    """Base class for area-under-curve-based metrics.

    All AUC-Metrics support continuous scorings, calculate the area under a curve function, and allow plotting this
    curve function. See the subclasses' documentation for a detailed explanation of the corresponding curve and metric.
    """

    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        self._plot = plot
        self._plot_store = plot_store

    def _auc(self, y_true: np.ndarray, y_score: Iterable[float], curve_function: Callable[[np.ndarray, np.ndarray], Any]) -> float:
        x, y, thresholds = curve_function(y_true, np.array(y_score))
        if "precision_recall" in curve_function.__name__:
            # swap x and y
            x, y = y, x
        area: float = auc(x, y)
        if self._plot:
            import matplotlib.pyplot as plt

            name = curve_function.__name__
            plt.plot(x, y, label=name, drawstyle="steps-post")
            # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.title(f"{name} | area = {area:.4f}")
            if self._plot_store:
                plt.savefig(f"fig-{name}.pdf")
            plt.show()
        return area

    def supports_continuous_scorings(self) -> bool:
        return True


class RangePrecisionRangeRecallAUC(AucMetric):
    """Computes the area under the precision recall curve when using the range-based precision and range-based
    recall metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    max_samples: int
        TimeEval uses a community implementation of the range-based precision and recall metrics, which is quite slow.
        To prevent long runtimes caused by scorings with high precision (many thresholds), just a specific amount of
        possible thresholds is sampled. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    r_alpha : float
        Weight of the existence reward for the range-based recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    plot : bool
    plot_store : bool
    name : str
        Custom name for this metric (e.g. including your parameter changes).


    .. rubric:: References

    .. [TatbulEtAl2018] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(
        self,
        max_samples: int = 50,
        r_alpha: float = 0.5,
        p_alpha: float = 0,
        cardinality: str = "reciprocal",
        bias: str = "flat",
        plot: bool = False,
        plot_store: bool = False,
        name: str = "RANGE_PR_AUC",
    ) -> None:
        super().__init__(plot, plot_store)
        self._max_samples = max_samples
        self._r_alpha = r_alpha
        self._p_alpha = p_alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, self._range_precision_recall_curve)

    def _range_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.unique(y_score)
        thresholds.sort()
        # The first precision and recall values are precision=class balance and recall=1.0, which corresponds to a
        # classifier that always predicts the positive class, independently of the threshold. This means that we can
        # skip the first threshold!
        p0 = y_true.sum() / len(y_true)
        r0 = 1.0
        thresholds = thresholds[1:]

        # sample thresholds
        n_thresholds = thresholds.shape[0]
        if n_thresholds > self._max_samples:
            every_nth = n_thresholds // (self._max_samples - 1)
            sampled_thresholds = thresholds[::every_nth]
            if thresholds[-1] == sampled_thresholds[-1]:
                thresholds = sampled_thresholds
            else:
                thresholds = np.r_[sampled_thresholds, thresholds[-1]]

        recalls = np.zeros_like(thresholds)
        precisions = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            y_pred = (y_score >= threshold).astype(np.int64)
            recalls[i] = ts_recall(y_true, y_pred, alpha=self._r_alpha, cardinality=self._cardinality, bias=self._bias)
            precisions[i] = ts_precision(y_true, y_pred, alpha=self._p_alpha, cardinality=self._cardinality, bias=self._bias)
        # first sort by recall, then by precision to break ties (important for noisy scorings)
        sorted_idx = np.lexsort((precisions * (-1), recalls))[::-1]
        return np.r_[p0, precisions[sorted_idx], 1], np.r_[r0, recalls[sorted_idx], 0], thresholds

    @property
    def name(self) -> str:
        return self._name


metric_auc_ptrt = RangePrecisionRangeRecallAUC(max_samples=50, r_alpha=0.5, p_alpha=0, cardinality="reciprocal", bias="flat")


def get_metrics(score, labels, slidingWindow=100, pred=None, version="opt", thre=250, anomaly_ratio=0.05) -> dict[str, float]:
    """Computes a reduced set of metrics compared to mTSBench for faster computation. Keep the important VUS_PR and AUC_ROC metrics."""
    # From mTSBench
    metrics_dict = {}

    # Threshold Independent
    grader = basic_metricor()
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    VUS_ROC, VUS_PR = compute_vus_score(labels.astype(int), score, slidingWindow, version, thre)

    # Threshold Dependent
    # if pred is None --> use the oracle threshold

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)

    metrics_dict["AUC-PR"] = AUC_PR
    metrics_dict["AUC-ROC"] = AUC_ROC
    metrics_dict["VUS-PR"] = VUS_PR
    metrics_dict["VUS-ROC"] = VUS_ROC

    metrics_dict["Standard-F1"] = PointF1
    metrics_dict["Event-based-F1"] = EventF1PA

    # adding other evaluation metrics

    # Identify the top anomaly candidates based on the anomaly ratio
    cutoff_percentile = 100 * (1 - anomaly_ratio * 1.5)  # Example: 1% anomaly => 99th percentile
    threshold = np.percentile(score, cutoff_percentile)
    predicted_anomalies = (score >= threshold).astype(int)  # Binary predictions
    # Compute metrics
    try:
        recall = metrics.recall_score(labels, predicted_anomalies)
        precision = metrics.precision_score(labels, predicted_anomalies, zero_division=0)
        # anom_scores =score.tolist()
        # # print(anom_scores)
        auc_ptrt = metric_auc_ptrt.score(labels, score)
    except ValueError as e:
        print(f"Error computing metrics : {e}")
        recall = 0
        precision = 0
        auc_ptrt = 0
    metrics_dict["Recall"] = recall
    metrics_dict["Precision"] = precision
    metrics_dict["AUC-PTRT"] = auc_ptrt
    return metrics_dict


def compute_vus_score(label, score, slidingWindow, version="opt", thre=250):
    if version == "opt_mem":
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_mem(
            labels_original=label, score=score, windowSize=slidingWindow, thre=thre
        )
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt(
            labels_original=label, score=score, windowSize=slidingWindow, thre=thre
        )
    return avg_auc_3d, avg_ap_3d


class basic_metricor:
    def __init__(
        self,
        a=1,
        probability=True,
        bias="flat",
    ):
        self.a = a
        self.probability = probability
        self.bias = bias
        self.eps = 1e-15

    def metric_ROC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def metric_PointF1(self, label, score, preds=None):
        if preds is None:
            precision, recall, thresholds = metrics.precision_recall_curve(label, score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 0.00001)
            F1 = np.max(f1_scores)
            threshold = thresholds[np.argmax(f1_scores)]
        else:
            Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            F1 = F[1]
        return F1

    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, score, preds=None):
        from sklearn.metrics import precision_score

        true_events = self._get_events(label)

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            EventF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                tp = np.sum([preds[start : end + 1].any() for start, end in true_events.values()])
                fn = len(true_events) - tp
                rec_e = tp / (tp + fn)
                prec_t = precision_score(label, preds)
                EventF1PA = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

                EventF1PA_scores.append(EventF1PA)

            EventF1PA_Threshold = thresholds[np.argmax(EventF1PA_scores)]
            EventF1PA1 = max(EventF1PA_scores)

        else:

            tp = np.sum([preds[start : end + 1].any() for start, end in true_events.values()])
            fn = len(true_events) - tp
            rec_e = tp / (tp + fn)
            prec_t = precision_score(label, preds)
            EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return EventF1PA1

    def range_convers_new(self, label):
        """
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        """
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        (anomaly_ends,) = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    # TPR_FPR_window
    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0] : seg[1] + 1] = labels_extended[seg[0] : seg[1] + 1] * pred[seg[0] : seg[1] + 1]
                    if (pred[seg[0] : (seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0] : seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0] : seg[1] + 1], pred[seg[0] : seg[1] + 1])
                    N_labels += np.sum(labels[seg[0] : seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = AUC_range

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0] : seg[1] + 1] = labels_extended[seg[0] : seg[1] + 1] * p[j][seg[0] : seg[1] + 1]
                    if (p[j][seg[0] : (seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0] : seg[1] + 1] = 1

                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0] : seg[1] + 1], p[j][seg[0] : seg[1] + 1])
                    N_labels += np.sum(labels[seg[0] : seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio

                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = AUC_range

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)
