"""
Run TS_UCF on the CATS datasets and save the AC@k  to results/impact_score_res.csv
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ts_ucf import plots
from ts_ucf.core import infer_window_size
from ts_ucf.ts_ucf import TS_UCF

logger = logging.getLogger("TS_UCF.impact_score_experiments")


def load_data(data_folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load CATS dataset.
    Returns scaled_data_train, scaled_data_test, labels test and metadata with the 'root_cause_chan' col."""
    data = pd.read_csv(os.path.join(data_folder, "data.csv"))
    split_train_test = int(1e6)
    # First col is timestamp and two last cols are y and category
    data_train = data.iloc[:split_train_test, 1:-2].to_numpy()
    data_test = data.iloc[split_train_test:, 1:-2].to_numpy()
    labels_test = data.iloc[split_train_test:, -2].to_numpy()
    scaler = MinMaxScaler((-1, 1))
    scaled_data_train = scaler.fit_transform(data_train)
    scaled_data_test = scaler.transform(data_test)

    metadata = pd.read_csv(os.path.join(data_folder, "metadata.csv"))
    # First col is timestamp so the indices are from 1 to 17
    metadata["root_cause_chan"] = metadata.root_cause.apply(lambda x: data.columns.tolist().index(x))
    metadata["affected_chan"] = metadata.affected.apply(lambda x: data.columns.tolist().index(x[2:-2]))

    return scaled_data_train, scaled_data_test, labels_test, metadata


def window_score_to_point_score(score: np.ndarray, L: int, N: int) -> np.ndarray:
    """Pad the scores to obtain a point by point score of length N and not a windows by windows score."""
    point_scores = np.repeat(score, L, axis=0)
    padding_length = N - len(point_scores)
    return np.pad(point_scores, ((0, padding_length), (0, 0)), mode="edge")


def run(
    detector, data_train, data_test, labels_test, n_min=1, n_max=10, idx_degree_metric=1, window_size=None, sigma=0.2
) -> tuple[np.ndarray, dict[str, str | float], np.ndarray]:
    """Run TS_UCF on the preprocessed data_train and data_test sets and compute the impact scores.
    Do not calculate the metrics to save time.
    Reutrns the padded point scores, the record without the metrics, the padded impact scores."""
    start = time.time()
    if window_size is None:
        ws = infer_window_size(data_train)
        L = ws
    else:
        L = window_size

    clf = TS_UCF(n_min=n_min, n_max=n_max, L=L, keep_internals=True)
    clf.fit(data_train)
    score = clf.decision_function(data_test)[-1]  # Multivariate result
    padded_point_scores = window_score_to_point_score(score, L, len(data_test))
    end = time.time()
    runtime = end - start
    logger.info("Runtime: %f s", runtime)

    start_impact = time.time()
    impact_scores = clf.impact_scores(sigma=sigma)
    padded_impact_scores = window_score_to_point_score(impact_scores, L, len(data_test))
    end_impact = time.time()
    runtime_impact = end_impact - start_impact
    logger.info("Impact Score Runtime: %f s", runtime_impact)

    start_metrics = time.time()
    # metrics = get_metrics(padded_point_scores[:, idx_degree_metric], labels_test)
    end_metrics = time.time()
    logger.debug("Metrics Runtime: %f s", end_metrics - start_metrics)
    record = {
        "data_file": "cats",
        "model": detector,
        "runtime": runtime,
        # **metrics
    }
    return padded_point_scores, record, padded_impact_scores


def compute_event_recall(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False) -> float:
    """Compute the event recall."""
    anos_seq = plots.find_ano_events(y_true)
    if (total_events := len(anos_seq)) == 0:
        return 1

    recalled_events = 0

    for start, end in anos_seq:
        if np.any(y_pred[start:end] == 1):
            recalled_events += 1

    if verbose:
        logger.info("Anomaly detected %d / %d", recalled_events, total_events)
    return recalled_events / total_events


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the point by point precision."""
    pos_pred = y_pred == 1
    true_positives = np.sum((y_true == 1) & pos_pred)
    false_positives = np.sum((y_true == 0) & pos_pred)

    total_positives = true_positives + false_positives

    if total_positives == 0:
        return 0.0

    return true_positives / total_positives


def compute_F1_score(precision: float, recall: float) -> float:
    """Compute the F1 score based on the given precision and recall."""
    sum_ = precision + recall
    if sum_ == 0:
        return 0
    return 2 * (precision * recall) / sum_


def get_th_opt(scores_ts_ucf: np.ndarray, labels_test: np.ndarray) -> float:
    """Compute the threshold that maximizes the F1 score using event recall and point by point precision."""
    max_f1 = 0
    th_opt = 0
    for th in sorted(np.unique(scores_ts_ucf[:, 1]))[::10]:  # ::10 to be faster
        curr_y_pred = (scores_ts_ucf[:, 1] > th).astype(int)
        curr_recall = compute_event_recall(labels_test, curr_y_pred)
        curr_precision = compute_precision(labels_test, curr_y_pred)
        curr_f1 = compute_F1_score(curr_precision, curr_recall)
        if curr_f1 > max_f1:
            th_opt = th
            max_f1 = curr_f1
    return th_opt


def compute_ACk(y_true: np.ndarray, y_pred: np.ndarray, root_causes: list, impact_scores: np.ndarray, k: int = 3, verbose: bool = False) -> float:
    """Compute the AC@k for a given k."""
    anos_seq = plots.find_ano_events(y_true)
    if len(anos_seq) == 0:
        return 1

    num_ano_detected = 0
    num_ac = 0
    for num_ano, (start, end) in enumerate(anos_seq):
        idx_ano = np.flatnonzero(y_pred[start:end])
        if len(idx_ano) > 0:
            impact_scores_ano = impact_scores[start:end][idx_ano]
            med_impact_scores_ano = np.median(np.unique(impact_scores_ano, axis=0), axis=0)
            sorted_impacts = np.argsort(med_impact_scores_ano)[::-1] + 1
            root_cause_pos = np.flatnonzero(sorted_impacts == root_causes[num_ano]) + 1
            num_ano_detected += 1
            if root_cause_pos <= k:
                num_ac += 1

    if verbose:
        logger.info("AC@k for k=%d = %d / %d = %.6g", k, num_ac, num_ano_detected, num_ac / num_ano_detected)

    return num_ac / num_ano_detected


# Main experiments
def main(data_folder: str):
    """Setup the logger, load the data, compute the TS_UCF scores and impact scores, find the optimal threshold, compute and save the AC@k for k in 1 to 5."""
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(os.path.dirname(exp_dir), "results")

    os.makedirs(res_dir, exist_ok=True)

    # Logger
    ts_ucf_logger = logging.getLogger("TS_UCF")
    ts_ucf_logger.setLevel(logging.DEBUG)
    if not ts_ucf_logger.handlers:
        # Stream handler for console output
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s.%(funcName)s (L%(lineno)d) - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        ts_ucf_logger.addHandler(handler)

        # File handler for log file
        log_file = os.path.join(res_dir, "ts_ucf_impact.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)  # Set the file handler to INFO level
        file_formatter = logging.Formatter("%(asctime)s - %(name)s.%(funcName)s (L%(lineno)d) - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        ts_ucf_logger.addHandler(file_handler)

    # Load data
    logger.info("Loading data...")
    scaled_data_train, scaled_data_test, labels_test, metadata = load_data(data_folder)

    # Run TS-UCF
    logger.info("Running TS_UCF...")
    scores_ts_ucf, _, impact_scores_ucf = run("TS_UCF", scaled_data_train, scaled_data_test, labels_test, window_size=500)

    # Compute optimal threshold and metrics
    logger.info("Computing optimal threshold...")
    th_opt = get_th_opt(scores_ts_ucf, labels_test)
    y_pred = (scores_ts_ucf[:, 1] > th_opt).astype(int)
    recall = compute_event_recall(labels_test, y_pred, verbose=True)
    precision = compute_precision(labels_test, y_pred)
    f1 = compute_F1_score(precision, recall)
    logger.info("Metrics: F1=%.6g - precision=%.6g - recall=%.6g - th=%.6g", f1, precision, recall, th_opt)

    # Compute ACks
    logger.info("Computing ACks...")
    acks = {}
    for k in range(1, 6):
        ack = compute_ACk(labels_test, y_pred, metadata["root_cause_chan"].tolist(), impact_scores_ucf, k=k, verbose=True)
        acks[k] = ack
    logger.info("ACKs = %s", acks)

    # Save results
    df = pd.DataFrame.from_dict(acks, orient="index", columns=["ACk"])
    df.index.name = "k"
    df = df.reset_index()
    res_path = os.path.join(res_dir, "impact_score_res.csv")
    df.to_csv(res_path, index=False)
    logger.info("Results saved in '%s'", res_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, required=True)

    args = parser.parse_args()
    main(args.datafolder)
