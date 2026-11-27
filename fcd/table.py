"""Table 1 of the paper and the Mean of each model, from per-example scores.

    python -m fcd table                                     # from the records shipped in records/
    python -m fcd table --runs runs --models qwen3-8b       # from your own runs (fcd run)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .data import GROUPS, HELDOUT, LABELS as TASK_LABELS
from .methods import LABELS, TABLE1
from .models import LABELS as MODEL_LABELS, MODELS
from .stats import paired_bootstrap, summarize

MAIN = tuple(t for ts in GROUPS.values() for t in ts)
TASKS = MAIN + HELDOUT
#: every difference is taken relative to plain sampling
BASE = "sampling"
#: records/ of a clone of the repository
RECORDS = Path(__file__).resolve().parents[1] / "records"
_HELD = {"heldout": HELDOUT}


def build(scores, clusters, models: Sequence[str] = tuple(MODELS), methods: Sequence[str] = TABLE1,
          base: str = BASE, n_boot: int = 10000, heldout: bool = True) -> Dict[str, dict]:
    """Scores and differences from ``base`` with 95% intervals of every method. With ``heldout``, also the average of
    the three ConFiQA subsets and its difference from ``base``."""
    rows = {}
    for m in methods:
        row = summarize(scores, m, models)
        row["vs_base"] = None if m == base else paired_bootstrap(scores, m, base, models, clusters, n_boot=n_boot)
        if heldout:
            held = summarize(scores, m, models, _HELD)
            row["heldout"] = {"tasks": held["tasks"], "mean": held["groups"]["heldout"]}
            row["heldout_vs_base"] = None if m == base else \
                paired_bootstrap(scores, m, base, models, clusters, _HELD, n_boot=n_boot)["heldout"]
        rows[m] = row
    return rows


def _diff(ci: Optional[dict]) -> str:
    if ci is None:
        return ""
    d = round(ci["delta"], 1)
    sign = "+" if d > 0 else "-" if d < 0 else ""
    return f" ({sign}{abs(d):.1f}{'*' if ci['significant'] else ''})"


def markdown(rows: Mapping[str, dict]) -> str:
    """Table 1 as Markdown. An asterisk marks a difference from sampling whose 95% interval excludes zero."""
    held = all("heldout" in r for r in rows.values())
    head = ["Method", *(TASK_LABELS[t] for t in GROUPS["counterfactual"]), "Counterfactual",
            *(TASK_LABELS[t] for t in GROUPS["ordinary"]), "Ordinary", "Mean"] + (["ConFiQA (held out)"] if held else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m, r in rows.items():
        vs = r["vs_base"] or {}
        cells = [LABELS.get(m, m), *(f"{r['tasks'][t]:.1f}" for t in GROUPS["counterfactual"]),
                 f"{r['groups']['counterfactual']:.1f}{_diff(vs.get('counterfactual'))}",
                 *(f"{r['tasks'][t]:.1f}" for t in GROUPS["ordinary"]),
                 f"{r['groups']['ordinary']:.1f}{_diff(vs.get('ordinary'))}", f"{r['mean']:.1f}{_diff(vs.get('mean'))}"]
        if held:
            cells.append(f"{r['heldout']['mean']:.1f}{_diff(r['heldout_vs_base'])}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_heldout(rows: Mapping[str, dict]) -> str:
    """The three ConFiQA subsets and their average."""
    head = ["Method", *(TASK_LABELS[t] for t in HELDOUT), "Average"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m, r in rows.items():
        h = r["heldout"]
        lines.append("| " + " | ".join([LABELS.get(m, m), *(f"{h['tasks'][t]:.1f}" for t in HELDOUT),
                                        f"{h['mean']:.1f}{_diff(r['heldout_vs_base'])}"]) + " |")
    return "\n".join(lines)


def markdown_per_model(rows: Mapping[str, dict], models: Sequence[str]) -> str:
    """Mean of the two group averages for each model."""
    head = ["Method", *(MODEL_LABELS.get(m, m) for m in models)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m, r in rows.items():
        lines.append("| " + " | ".join([LABELS.get(m, m), *(f"{r['per_model'][x]:.1f}" for x in models)]) + " |")
    return "\n".join(lines)


def intervals(rows: Mapping[str, dict]) -> List[str]:
    """One line per method: the difference from sampling and its 95% interval for each group, Mean and ConFiQA."""
    out = []
    for m, r in rows.items():
        if not r["vs_base"]:
            continue
        cis = dict(r["vs_base"])
        if r.get("heldout_vs_base"):
            cis["confiqa"] = r["heldout_vs_base"]
        parts = [f"{g} {c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]{'*' if c['significant'] else ''}"
                 for g, c in cis.items()]
        out.append(f"{LABELS.get(m, m):21s} " + "   ".join(parts))
    return out


def from_records(root=RECORDS, models: Sequence[str] = tuple(MODELS), methods: Sequence[str] = TABLE1,
                 n_boot: int = 10000) -> Dict[str, dict]:
    from .records import load_paper_records

    scores, clusters = load_paper_records(root, models, TASKS, _with_base(methods))
    return build(scores, clusters, models, methods, n_boot=n_boot)


def from_runs(runs, models: Sequence[str], methods: Sequence[str] = TABLE1, seeds: Sequence[int] = (0, 1, 2),
              splits=RECORDS / "splits.json", tasks: Sequence[str] = TASKS, n_boot: int = 10000) -> Dict[str, dict]:
    """Table 1 from runs. Pass ``tasks=MAIN`` when the ConFiQA subsets were not run."""
    from .records import load_runs, load_splits

    scores, clusters, _ = load_runs(runs, models, tasks, _with_base(methods), seeds, load_splits(splits))
    return build(scores, clusters, models, methods, n_boot=n_boot, heldout=all(t in tasks for t in HELDOUT))


def _with_base(methods: Sequence[str]) -> List[str]:
    return list(dict.fromkeys([BASE, *methods]))
