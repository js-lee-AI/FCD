"""Every run behind Table 1: sampling, the four guidance baselines and FCD on the five passage tasks and the three
ConFiQA subsets with three seeds, for each model (GPU).

    python experiments/run_main.py --models qwen3-8b --out runs
    python experiments/run_main.py --out runs                       # all six models, one after another
    python experiments/table1_main.py --runs runs --models qwen3-8b

A run is skipped when its summary exists, so the script resumes after an interruption.
"""

import argparse

from fcd.data import TASKS
from fcd.methods import TABLE1
from fcd.models import MODELS
from fcd.run import run_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--methods", nargs="+", default=list(TABLE1), choices=list(TABLE1))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    device = int(a.device) if a.device.isdigit() else a.device
    for model in a.models:
        run_model(model, tasks=a.tasks, methods=a.methods, seeds=a.seeds, out=a.out, device=device)


if __name__ == "__main__":
    main()
