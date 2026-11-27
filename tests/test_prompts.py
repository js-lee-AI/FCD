"""Each matched view changes exactly one factor of the full prompt."""

import pytest

from fcd import Example, build_prompt, views
from fcd.demo import COUNTERFACTUAL
from fcd.prompts import KINDS


def parts(text):
    """(directive and format, passage block, question block) of a prompt with a passage."""
    return tuple(text.split("\n\n", 2))


def test_the_views_change_one_factor_each():
    v = views(COUNTERFACTUAL)
    full, d, e = parts(v["full"]), parts(v["directive_view"]), parts(v["evidence_view"])
    assert full[0] == "Answer the question using the passage. Reply with only the answer, a few words at most."
    assert d[0] == ("Answer the question using your own knowledge and ignore the passage. "
                    "Reply with only the answer, a few words at most.")
    assert d[1:] == full[1:]
    assert e[0] == full[0] and e[2] == full[2]
    assert e[1] == f"Passage: {COUNTERFACTUAL.partner_passage}" != full[1]


def test_question_only_drops_the_passage_and_rewrites_the_directive():
    q = build_prompt(COUNTERFACTUAL, "question_only")
    assert q == ("Answer the question. Reply with only the answer, a few words at most.\n\n"
                 "Question: who wrote papa got a brand new bag")


def test_multiple_choice_and_several_passages():
    ex = Example("Which would form a fossil?", "p", options=("a rock", "a bone"), partner_passage="other")
    assert build_prompt(ex, "full").startswith("Answer the multiple-choice question according to the passage. "
                                               "Reply with only the letter of the correct option.")
    assert build_prompt(ex, "full").endswith("Question: Which would form a fossil?\n\nA. a rock\nB. a bone")
    assert build_prompt(ex, "directive_view").startswith(
        "Answer the multiple-choice question using your own knowledge and ignore the passage.")
    hp = Example("q", "T1: s1\n\nT2: s2", multi_passage=True, partner_passage="T3: s3")
    assert build_prompt(hp, "evidence_view") == ("Answer the question using the passages. Reply with only the answer, "
                                                 "a few words at most.\n\nPassages: T3: s3\n\nQuestion: q")


def test_the_evidence_view_needs_another_passage():
    with pytest.raises(ValueError):
        build_prompt(Example("q", "p"), "evidence_view")
    with pytest.raises(ValueError):
        build_prompt(Example("q", "p"), "unknown")
    assert set(KINDS) == {"full", "directive_view", "evidence_view", "question_only"}
