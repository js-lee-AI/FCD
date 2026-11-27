"""The decoders of the main comparison (paper Table 1), with the settings selected on the development split.

A decoder samples from a score computed at every step from the next-token log-probabilities under its prompts: the full
prompt x and, for a contrastive decoder, one or more views x^(k) of it. With l = log p(y | x) and l_k = log p(y | x^(k)),
guidance scores a token by

    l + sum_k gamma_k (l - l_k).

FCD uses two matched views, the directive view and the evidence view, each with its own strength gamma_D and gamma_E.
CAD, Random-passage CD and Instructive Decoding are the case of one view and one strength. AdaCAD scales the strength
at every step by the Jensen-Shannon divergence of the two distributions. Temperature and truncation are applied to the
score afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    views: Tuple[str, ...] = ()          # prompt kinds contrasted with the full prompt, see fcd.prompts
    rule: str = "plain"                  # plain, guidance or adacad
    strengths: Tuple[float, ...] = ()    # one strength per view

    def __post_init__(self) -> None:
        if len(self.strengths) != len(self.views):
            raise ValueError(f"{self.name}: {len(self.views)} views but {len(self.strengths)} strengths")

    def with_strengths(self, *strengths: float) -> "Method":
        return replace(self, strengths=tuple(float(s) for s in strengths))

    @property
    def streams(self) -> Tuple[str, ...]:
        return ("full", *self.views)


#: the strengths of FCD selected on the development split, one pair for all models and tasks
GAMMA_D, GAMMA_E = 1.0, 0.5

METHODS: Dict[str, Method] = {m.name: m for m in [
    Method("sampling", "Sampling"),
    Method("cad", "CAD", ("question_only",), "guidance", (1.0,)),
    # gamma is scaled at every step by the Jensen-Shannon divergence of the two distributions, in bits
    Method("adacad", "AdaCAD", ("question_only",), "adacad", (1.0,)),
    Method("rpcd", "Random-passage CD", ("evidence_view",), "guidance", (1.0,)),
    Method("id", "Instructive Decoding", ("directive_view",), "guidance", (2.0,)),
    Method("fcd", "FCD", ("directive_view", "evidence_view"), "guidance", (GAMMA_D, GAMMA_E)),
]}

#: rows of paper Table 1, in order
TABLE1 = ("sampling", "cad", "adacad", "rpcd", "id", "fcd")
LABELS = {k: m.label for k, m in METHODS.items()}


def get_method(name: str, strengths: Optional[Sequence[float]] = None) -> Method:
    try:
        m = METHODS[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(METHODS)}") from None
    return m if strengths is None else m.with_strengths(*strengths)


def fcd_method(gamma_d: float = GAMMA_D, gamma_e: float = GAMMA_E) -> Method:
    """FCD with the given strengths. ``fcd_method(0, 0)`` is plain sampling, ``fcd_method(g, 0)`` Instructive Decoding
    with strength g and ``fcd_method(0, g)`` Random-passage CD with strength g."""
    return METHODS["fcd"].with_strengths(gamma_d, gamma_e)
