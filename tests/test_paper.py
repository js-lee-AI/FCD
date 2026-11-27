"""Table 1, the per-model Means, the ConFiQA subsets and the comparison of FCD with every baseline, rebuilt from the
per-example scores in records/."""

import pytest

np = pytest.importorskip("numpy")

from fcd import table  # noqa: E402
from fcd.records import load_paper_records, load_splits  # noqa: E402
from fcd.stats import paired_bootstrap  # noqa: E402

MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "llama3.1-8b", "mistral-7b"]

# NQ-Swap, FaithEval, counterfactual avg., NQ-Orig., SQuAD, HotpotQA, ordinary avg., Mean, ConFiQA
TABLE1 = {
    "sampling": (56.9, 63.6, 60.3, 76.0, 84.5, 63.5, 74.7, 67.5, 63.1),
    "cad": (62.4, 73.7, 68.1, 72.5, 84.6, 60.8, 72.6, 70.3, 66.6),
    "adacad": (61.4, 65.5, 63.4, 75.6, 85.5, 63.2, 74.8, 69.1, 65.7),
    "rpcd": (61.0, 71.1, 66.1, 75.4, 85.3, 62.5, 74.4, 70.2, 67.8),
    "id": (64.3, 67.8, 66.0, 74.8, 85.6, 63.6, 74.7, 70.4, 69.4),
    "fcd": (64.9, 70.0, 67.4, 76.3, 86.4, 63.8, 75.5, 71.5, 70.3),
}
# difference from sampling on counterfactual, ordinary, Mean and ConFiQA, and whether its 95% interval excludes zero
DIFFS = {
    "cad": ((7.8, True), (-2.1, True), (2.9, True), (3.5, True)),
    "adacad": ((3.2, True), (0.1, False), (1.6, True), (2.5, True)),
    "rpcd": ((5.8, True), (-0.3, False), (2.8, True), (4.6, True)),
    "id": ((5.8, True), (0.0, False), (2.9, True), (6.3, True)),
    "fcd": ((7.2, True), (0.8, True), (4.0, True), (7.1, True)),
}
PER_MODEL = {
    "sampling": (65.1, 68.1, 72.2, 72.2, 69.5, 57.7), "cad": (65.3, 70.6, 76.2, 75.5, 72.3, 62.3),
    "adacad": (65.9, 69.4, 73.3, 73.2, 71.9, 60.9), "rpcd": (65.8, 70.4, 74.3, 74.6, 71.2, 65.1),
    "id": (62.9, 71.9, 75.5, 75.4, 71.4, 65.1), "fcd": (65.5, 71.9, 75.5, 75.3, 74.1, 66.5),
}
# ConFiQA QA, MR and MC
HELDOUT = {"sampling": (71.6, 61.9, 55.9), "fcd": (78.1, 68.5, 64.2)}
# FCD minus each baseline on Mean (paper Table 10)
FCD_VS = {"cad": 1.1, "adacad": 2.4, "rpcd": 1.2, "id": 1.1}


@pytest.fixture(scope="module")
def rows():
    return table.from_records(table.RECORDS, MODELS)


def test_splits():
    splits = load_splits(table.RECORDS / "splits.json")
    assert {t: len(s["scored"]) for t, s in splits.items()} == \
        {"nqswap": 724, "faitheval": 800, "nqorig": 724, "squad": 724, "hotpotqa": 800,
         "confiqa_qa": 800, "confiqa_mr": 800, "confiqa_mc": 800}
    for s in splits.values():
        assert len(s["dev"]) == 200 and len(s["test"]) == 800 and len(s["clusters"]) == len(s["scored"])
        assert set(s["scored"]) <= set(s["test"]) and not set(s["dev"]) & set(s["test"])


@pytest.mark.parametrize("method", list(TABLE1))
def test_table1(rows, method):
    r = rows[method]
    got = (r["tasks"]["nqswap"], r["tasks"]["faitheval"], r["groups"]["counterfactual"], r["tasks"]["nqorig"],
           r["tasks"]["squad"], r["tasks"]["hotpotqa"], r["groups"]["ordinary"], r["mean"], r["heldout"]["mean"])
    assert [round(x, 1) for x in got] == list(TABLE1[method])
    assert [round(r["per_model"][m], 1) for m in MODELS] == list(PER_MODEL[method])


@pytest.mark.parametrize("method", list(DIFFS))
def test_differences_from_sampling(rows, method):
    r = rows[method]
    cis = (r["vs_base"]["counterfactual"], r["vs_base"]["ordinary"], r["vs_base"]["mean"], r["heldout_vs_base"])
    for ci, (delta, significant) in zip(cis, DIFFS[method]):
        assert round(ci["delta"], 1) == delta
        assert ci["significant"] is significant


@pytest.mark.parametrize("method", list(HELDOUT))
def test_confiqa_subsets(rows, method):
    h = rows[method]["heldout"]["tasks"]
    assert [round(h[t], 1) for t in ("confiqa_qa", "confiqa_mr", "confiqa_mc")] == list(HELDOUT[method])


def test_fcd_has_the_highest_mean_and_beats_every_baseline_on_it():
    scores, clusters = load_paper_records(table.RECORDS, MODELS, table.MAIN, list(TABLE1))
    for base, delta in FCD_VS.items():
        ci = paired_bootstrap(scores, "fcd", base, MODELS, clusters)["mean"]
        assert round(ci["delta"], 1) == delta and ci["lo"] > 0, base
