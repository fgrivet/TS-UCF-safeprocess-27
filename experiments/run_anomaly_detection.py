"""
Run TS_UCF on all mTSBench datasets for a given window size and save the metrics to results/TS_UCF_{ws}_evaluation_results.csv
"""

import argparse
import glob
import logging
import multiprocessing
import os
import time
from typing import Literal

import numpy as np
import pandas as pd
from filelock import FileLock
from mtsbench_metrics import get_metrics
from sklearn.preprocessing import MinMaxScaler

from ts_ucf.core import infer_window_size
from ts_ucf.ts_ucf import TS_UCF

logger = logging.getLogger("TS_UCF.mTSBench_experiments")


# Utils: get data and compute metrics
def get_data_labels(df: pd.DataFrame, need_label: bool) -> tuple[np.ndarray, None | np.ndarray]:
    """Extract data and label from a dataframe.
    Since not all datasets necessarily have labels for the training sets,
    you need to set `need_label = False` so that the `label` column is not required.
    """
    if "is_anomaly" in df.columns:
        data = df.iloc[:, 1:-1].values.astype(float)  # excluede first (time stamp) and last (label) columns
        label = df["is_anomaly"].astype(int).to_numpy()
    else:
        if need_label:
            raise ValueError("label needed and 'is_anomaly' not in df columns")
        data = df.iloc[:, 1:].values.astype(float)
        label = None
    return data, label


# Run TS-UCF with train and test data and run detector based on a dataset name
def run_TS_UCF(data_train, data_test, periodicity=1, degree=2, window_size=None) -> np.ndarray:
    """Run TS_UCF for a given degree on the preprocessed data_train and data_test sets with a fixed window size (infered if None).
    Returns a padded multivariate score so that it is calculated point by point rather than window by window."""
    scaler = MinMaxScaler((-1, 1))
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    if window_size is None:
        ws = infer_window_size(data_train_scaled)
        L = ws
    else:
        L = window_size

    clf = TS_UCF(n_min=degree, n_max=degree, L=L)
    clf.fit(data_train_scaled)
    score = clf.decision_function(data_test_scaled)[-1].ravel()  # Multivariate result
    point_scores = np.repeat(score, L)
    padding_length = len(data_test_scaled) - len(point_scores)
    padded_point_scores = np.pad(point_scores, (0, padding_length), mode="edge")
    return padded_point_scores


def run_detector_dataset(
    detector: str, folder_name: str, file_name: str, data_test: np.ndarray, labels_test: np.ndarray, data_train: np.ndarray, **kwargs
) -> tuple[np.ndarray, dict[str, str | float]]:
    """Run TS_UCF on the preprocessed data_train and data_test sets and compute the metrics.
    Returns the padded multivariate score and a record containing all the information on the dataset, model, runtime, and metrics."""
    start = time.time()
    output = run_TS_UCF(data_train, data_test, **kwargs)
    end = time.time()
    runtime = end - start
    logger.info("Runtime for '%s': %f s", file_name, runtime)

    start_metrics = time.time()
    metrics = get_metrics(output, labels_test)
    end_metrics = time.time()
    metrics_runtime = end_metrics - start_metrics
    logger.info("Metrics Runtime for '%s': %f s", file_name, metrics_runtime)

    record = {
        "data_folder": folder_name,
        "data_file": file_name,
        "model": detector,
        "runtime": runtime,
        "metrics_runtime": metrics_runtime,
        **metrics,
    }
    return output, record


# Write results and process file in parallel
def write_res(log_file, record):
    """Write results to the log_file."""
    with FileLock(log_file + ".lock"):
        df_log = pd.DataFrame([record])
        write_header = not os.path.exists(log_file)
        df_log.to_csv(log_file, mode="a", header=write_header, index=False)


def process_file(args):
    """For a given file: load and normalize the data, compute the TS_UCF results and metrics, save all the information to the results file.
    Input args: num_file, filepath, window_size, res_file_path."""
    num_file, fp, window_size, res_file_ts_ucf = args
    # Get folder name and file name for result saving
    fp_split = fp.rsplit(os.sep, 2)
    if len(fp_split) != 3:
        logger.error("File path '%s' is invalid.", fp)
        return
    _, folder_name, file_name = fp_split

    logger.info("Treating file %d: '%s' from folder '%s'", num_file, file_name, folder_name)

    try:
        # Load data
        df_test = pd.read_csv(fp)
        df_train = pd.read_csv(fp.replace("_test.csv", "_train.csv"))

        # Format data to numpy arrays
        data_train, _ = get_data_labels(df_train, False)
        data_test, labels_test = get_data_labels(df_test, True)

        # Compute scores and metrics
        _, metrics_ts_ucf = run_detector_dataset("TS_UCF", folder_name, file_name, data_test, labels_test, data_train, window_size=window_size)
        # Write results
        write_res(res_file_ts_ucf, metrics_ts_ucf)
    except:
        logger.exception("Error during processing of file '%s' from folder '%s'", file_name, folder_name)


# Main experiments
def main(data_folder: str, window_size: int | Literal["auto"], njobs: int = -1):
    """Setup the logger, find all the mTSBench datasets, run the experiment for each dataset (in parallel if njobs != 1), and save the results."""
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
        log_file = os.path.join(res_dir, f"ts_ucf_{window_size}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)  # Set the file handler to INFO level
        file_formatter = logging.Formatter("%(asctime)s - %(name)s.%(funcName)s (L%(lineno)d) - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        ts_ucf_logger.addHandler(file_handler)

    # Get all files to test
    file_paths = glob.glob(f"{data_folder}/*/*_test.csv")

    res_file_ts_ucf = os.path.join(res_dir, f"TS_UCF_{window_size}_evaluation_results.csv")

    if window_size == "auto":
        window_size = None  # type: ignore

    if njobs == 1:
        for i, fp in enumerate(file_paths):
            process_file((i, fp, window_size, res_file_ts_ucf))

    else:
        # Create a pool of workers
        with multiprocessing.Pool(processes=njobs) as pool:
            # Prepare arguments for each file
            args_list = [(i, fp, window_size, res_file_ts_ucf) for i, fp in enumerate(file_paths)]
            # Map the process_file function to the list of arguments
            pool.map(process_file, args_list)


# Argparse and launch program
def parse_positive_or_auto(value):
    """Returns an integer greater than 0 or 'auto'."""
    if value == "auto":
        return value
    try:
        int_val = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not valid as either an integer or 'auto'") from exc
    if int_val <= 0:
        raise argparse.ArgumentTypeError("The integer must be strictly greater than 0")
    return int_val


def parse_positive_minus_1(value):
    """Returns an integer greater than 0 or equal to -1."""
    try:
        int_val = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not valid as either an integer or 'auto'") from exc

    if int_val != -1 and int_val <= 0:
        raise argparse.ArgumentTypeError("The integer must be strictly greater than 0 or -1")
    return int_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, required=True)
    parser.add_argument("--ws", type=parse_positive_or_auto, required=True)
    parser.add_argument("--njobs", type=parse_positive_minus_1, required=False, default=-1)

    args = parser.parse_args()
    main(args.datafolder, args.ws, args.njobs)
