# -*- coding: utf-8 -*-
"""Wrapper script to run evaluation and monitor progress"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Output directory
output_dir = Path("rag_eval/evals/experiments")
status_file = output_dir / "run_status.txt"

def main():
    # Clear previous status
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, 'w') as f:
        f.write(f"[{datetime.now()}] Starting evaluation...\n")

    # Run the evaluation script
    cmd = [
        sys.executable, "-u",
        "rag_eval/evaluate_parallel_v2_fixed.py",
        "--testset", "rag_eval/evals/datasets/test_5q.json",
        "--workers", "2",
        "--batch-size", "2",
        "--output", "rag_eval/evals/experiments/test_5q_fixed.json"
    ]

    print("Starting evaluation with command:")
    print(" ".join(cmd))
    print()

    # Run with output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Monitor output
    last_status = time.time()
    while True:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue

        # Print to console
        print(line.rstrip())
        sys.stdout.flush()

        # Update status file every 5 seconds
        if time.time() - last_status > 5:
            with open(status_file, 'a') as f:
                f.write(f"[{datetime.now()}] {line.rstrip()}\n")
            last_status = time.time()

    # Final status
    returncode = process.wait()
    with open(status_file, 'a') as f:
        f.write(f"\n[{datetime.now()}] Evaluation complete. Return code: {returncode}\n")

    print(f"\nEvaluation complete. Return code: {returncode}")
    print(f"Status saved to: {status_file}")

    return returncode

if __name__ == "__main__":
    sys.exit(main())
