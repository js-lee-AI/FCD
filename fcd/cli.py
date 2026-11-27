"""The `fcd` command. `python -m fcd` runs the same thing."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .data import TASKS
from .methods import GAMMA_D, GAMMA_E, METHODS, TABLE1
from .models import MODELS

#: a passage that says nothing about most questions, the default evidence view of `fcd ask`
DEFAULT_OTHER_PASSAGE = ("The essential nature of the atomic nucleus was established with the discovery of the neutron "
                         "by James Chadwick in 1932 and the determination that it was a new elementary particle, "
                         "distinct from the proton.")


def _demo(args) -> int:
    from .demo import main

    main()
    return 0


def _table(args) -> int:
    from pathlib import Path

    from . import table

    models = args.models or list(MODELS)
    if not (Path(args.records) / "splits.json").exists():
        raise SystemExit(f"no splits.json in {args.records}; pass --records with the records/ directory of a clone "
                         "of the repository")
    if args.runs:
        try:
            rows = table.from_runs(args.runs, models, args.methods, args.seeds,
                                   splits=Path(args.records) / "splits.json", tasks=args.tasks, n_boot=args.boot)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(f"{e}\nPass --models, --methods, --seeds and --tasks to match the runs in {args.runs}; "
                             "sampling is always needed, since differences are taken relative to it.")
    else:
        rows = table.from_records(args.records, models, args.methods, n_boot=args.boot)
    print(table.markdown(rows))
    print("\nScores averaged over models and seeds: F1, or accuracy for FaithEval. Mean averages the two groups and "
          "ConFiQA its three subsets. Parentheses give the difference from sampling, * when its 95% interval "
          "excludes zero.")
    if args.per_model:
        print()
        print(table.markdown_per_model(rows, models))
    if args.heldout and all("heldout" in r for r in rows.values()):
        print()
        print(table.markdown_heldout(rows))
    if args.intervals:
        print("\nDifference from sampling with 95% paired cluster-bootstrap intervals:")
        print("\n".join(table.intervals(rows)))
    return 0


def _run(args) -> int:
    from .run import run_model

    run_model(args.model, tasks=args.tasks, methods=args.methods, seeds=args.seeds, out=args.out, split=args.split,
              device=args.device, limit=args.limit, max_batch=args.max_batch)
    return 0


def _ask(args) -> int:
    from .example import Example
    from .hf import Decoder
    from .methods import fcd_method

    ex = Example(question=args.question, passage=args.passage, options=tuple(args.option or ()),
                 partner_passage=args.other_passage)
    dec = Decoder.from_pretrained(args.model, device=args.device)
    rows = [("sampling", "sampling"), (f"FCD (gamma_D={args.gamma_d:g}, gamma_E={args.gamma_e:g})",
                                       fcd_method(args.gamma_d, args.gamma_e))]
    for label, method in rows:
        print(f"{label:32s} {dec.generate([ex], method, seed=args.seed).texts[0]!r}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        prog="fcd", description="Factorized contrastive decoding: contrast the full prompt with a directive view and "
                                "an evidence view, each with its own strength.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="the views and the recorded answers of the paper's Figure 1, without a model")
    demo.set_defaults(func=_demo)

    tab = sub.add_parser("table", help="Table 1 from the shipped records or from your runs (CPU)")
    tab.add_argument("--records", default=None, help="records directory (default: records/ of the repository)")
    tab.add_argument("--runs", default=None, help="a directory written by `fcd run` instead of the records")
    tab.add_argument("--models", nargs="+", default=None, help="models to average over (default: all six)")
    tab.add_argument("--methods", nargs="+", default=list(TABLE1), choices=list(TABLE1))
    tab.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds of the runs")
    tab.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS),
                     help="tasks of the runs (default: all eight; ConFiQA is shown when its three subsets are given)")
    tab.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    tab.add_argument("--per-model", action="store_true", help="also print the Mean of each model")
    tab.add_argument("--heldout", action="store_true", help="also print the three ConFiQA subsets")
    tab.add_argument("--intervals", action="store_true", help="also print the 95%% intervals")
    tab.set_defaults(func=_table)

    run = sub.add_parser("run", help="run the decoders over the tasks with one model (GPU)")
    run.add_argument("--model", required=True, help=f"a short name ({', '.join(MODELS)}) or a Hugging Face id")
    run.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    run.add_argument("--methods", nargs="+", default=list(TABLE1), choices=list(METHODS), metavar="METHOD",
                     help=f"default: the rows of Table 1 ({' '.join(TABLE1)})")
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--split", default="test", choices=["test", "dev"])
    run.add_argument("--out", default="runs")
    run.add_argument("--device", default=0, help="CUDA device index or name")
    run.add_argument("--limit", type=int, default=None, help="only the first N examples of each split")
    run.add_argument("--max-batch", type=int, default=64)
    run.set_defaults(func=_run)

    ask = sub.add_parser("ask", help="answer one question about a passage with sampling and with FCD (GPU)")
    ask.add_argument("--model", default="qwen3-8b")
    ask.add_argument("--passage", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--other-passage", default=DEFAULT_OTHER_PASSAGE,
                     help="the passage of the evidence view, one that says nothing about the question "
                          "(default: a sentence about the atomic nucleus)")
    ask.add_argument("--option", action="append", help="an option of a multiple-choice question, repeated in order")
    ask.add_argument("--gamma-d", type=float, default=GAMMA_D, help="strength of the directive view")
    ask.add_argument("--gamma-e", type=float, default=GAMMA_E, help="strength of the evidence view")
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--device", default=0)
    ask.set_defaults(func=_ask)

    args = parser.parse_args(argv)
    if args.command == "table" and args.records is None:
        from .table import RECORDS

        args.records = RECORDS
    if getattr(args, "device", None) is not None and str(args.device).isdigit():
        args.device = int(args.device)
    return args.func(args)
