"""Prompts of the paper (Appendix A). Each is a single user turn, rendered later with the model's chat template.

A prompt has four factors: the directive (what to do with the passage), the format sentence, the evidence (the passage)
and the question. A matched view replaces exactly one factor and keeps the other three.

Kinds:
    full            the task prompt, used by sampling, by FCD and by every contrastive decoder as the first stream
    directive_view  the full prompt with the directive replaced by its opposite, "... using your own knowledge and
                    ignore the passage": FCD's view of the directive, and the contrast of Instructive Decoding
    evidence_view   the full prompt with the passage of another example: FCD's view of the evidence, and the contrast
                    of Random-passage CD
    question_only   the question without the passage, whose directive no longer mentions it, so two factors change at
                    once: the contrast of CAD and AdaCAD
"""

from __future__ import annotations

from .example import Example

KINDS = ("full", "directive_view", "evidence_view", "question_only")

_FORMAT = "Reply with only the answer, a few words at most."
_FORMAT_MC = "Reply with only the letter of the correct option."


def directive(kind: str, noun: str, mc: bool) -> str:
    """The directive of a prompt kind, without the format sentence."""
    task = "Answer the multiple-choice question" if mc else "Answer the question"
    if kind in ("full", "evidence_view"):
        return f"{task} according to the {noun}." if mc else f"{task} using the {noun}."
    if kind == "directive_view":
        return f"{task} using your own knowledge and ignore the {noun}."
    if kind == "question_only":
        return f"{task}."
    raise ValueError(f"unknown prompt kind {kind!r}; expected one of {KINDS}")


def build_prompt(example: Example, kind: str) -> str:
    """Prompt text, a single user turn, for one example and one prompt kind."""
    mc = example.is_multiple_choice
    noun = "passages" if example.multi_passage else "passage"
    question = f"Question: {example.question}"
    if mc:
        question += f"\n\n{example.options_block()}"
    instruction = f"{directive(kind, noun, mc)} {_FORMAT_MC if mc else _FORMAT}"
    if kind == "question_only":
        return f"{instruction}\n\n{question}"
    if kind == "evidence_view":
        if example.partner_passage is None:
            raise ValueError("the evidence view needs Example.partner_passage, the passage of another example")
        passage = example.partner_passage
    else:
        passage = example.passage
    return f"{instruction}\n\n{noun.capitalize()}: {passage}\n\n{question}"


def views(example: Example) -> dict:
    """The full prompt and its two matched views, by kind."""
    return {k: build_prompt(example, k) for k in ("full", "directive_view", "evidence_view")}
