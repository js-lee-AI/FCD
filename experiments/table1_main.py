"""Table 1 of the paper, the Mean of each model and the ConFiQA subsets, from the per-example scores in records/ or
from your runs. CPU.

    python experiments/table1_main.py                                   # the records behind the paper
    python experiments/table1_main.py --runs runs --models qwen3-8b     # runs written by experiments/run_main.py

With the records, the tables match the paper: scores averaged over six models and three seeds on the scored test
examples, and differences from sampling with their significance (95% paired cluster-bootstrap intervals, 10,000
resamples).
"""

import argparse

from fcd import table
from fcd.models import MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="a directory written by experiments/run_main.py or `fcd run`")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()

    rows = table.from_runs(a.runs, a.models, seeds=a.seeds) if a.runs else table.from_records(models=a.models)
    print("Table 1: test scores averaged over the models (F1, accuracy for FaithEval)\n")
    print(table.markdown(rows))
    print("\n* the 95% interval of the difference from sampling excludes zero\n")
    print("Mean of the counterfactual and ordinary averages for each model\n")
    print(table.markdown_per_model(rows, a.models))
    print("\nThe three ConFiQA subsets, held out from the choice of every setting\n")
    print(table.markdown_heldout(rows))
    print("\nDifferences from sampling with 95% intervals\n")
    print("\n".join(table.intervals(rows)))


if __name__ == "__main__":
    main()
