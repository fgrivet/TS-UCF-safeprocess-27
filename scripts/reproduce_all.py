"""Runs all the scripts needed to download the data
and reproduce the results and figures from the paper."""

import os
import subprocess

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    data_mtsbench_dir = os.path.join(code_dir, "data", "mTSBench")

    # Download data
    subprocess.run("python scripts/download_data.py", shell=True, cwd=code_dir, check=True)
    # Run anomaly detection on mTSBench for different window sizes
    for ws in [100, 250, 500, 1000, "auto"]:
        subprocess.run(
            f"python experiments/run_anomaly_detection.py --datafolder {data_mtsbench_dir} --ws {ws}", shell=True, cwd=code_dir, check=True
        )
    # Run impact score experiments on the CATS dataset
    subprocess.run(
        "python experiments/run_impact_score.py --datafolder data/controlled-anomalies-time-series-dataset", shell=True, cwd=code_dir, check=True
    )
    # Generate the figures of the article
    subprocess.run("jupyter execute notebooks/02_figures.ipynb", shell=True, cwd=code_dir, check=True)
