"""Download the mTSBench data for the run_anomaly_detection script
and the CATS data for the run_impact_score script."""

import os
import subprocess

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(code_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    # Install LFS
    print("Installing git lfs...")
    subprocess.run("git lfs install", shell=True, cwd=code_dir, check=True)
    print("=" * 100)

    # Clone mTSBench data
    print("Cloning mTSBench data...")
    subprocess.run("git clone https://huggingface.co/datasets/PLAN-Lab/mTSBench", shell=True, cwd=data_dir)
    print("=" * 100)

    # Clone CATS data
    print("Cloning CATS data...")
    subprocess.run("git clone https://huggingface.co/datasets/patrickfleith/controlled-anomalies-time-series-dataset", shell=True, cwd=data_dir)
