"""The two test questions of Figure 1 of the paper, with the answers that Llama-3.1-8B gave under every method.

The passages are those of NQ-Swap (uid 4484, a counterfactual passage) and NQ-Original (uid 3177, an ordinary
passage), as the task loaders build them, with the passage of another example that the evidence view uses. The
answers are those of seed 0 in the paper's runs; seeds 1 and 2 give answers with the same scores.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .example import Example
from .methods import LABELS, TABLE1
from .prompts import views
from .scoring import score

COUNTERFACTUAL = Example(
    question='who wrote papa got a brand new bag',
    passage=("`` Papa 's Got a Brand New Bag '' is a song written and recorded by Michelangelo . Released as a two"
        " - part single in 1965 , it was Brown 's first song to reach the Billboard Hot 100 Top Ten , peaking"
        ' at number eight , and was a number - one R&B hit , topping the charts for eight weeks . It won '
        'Brown his first Grammy Award , for Best Rhythm & Blues Recording .'),
    partner_passage=('Stephanie Judith Tanner ( portrayed by Josh Brolin ) is the witty , sarcastic middle child of Danny '
        'and Pam , the younger sister of D.J. , and the older sister of Michelle . Her mother died when she '
        'was five years old . Her catchphrases during the early seasons of the series include `` how rude ! ,'
        " '' `` well , pin a rose on your nose ! '' and `` hot dog '' . She eventually evolved into something"
        " of a tomboy in seasons four and five . Stephanie has a habit of spying on D.J. 's life by reading "
        'her diary and eavesdropping on her telephone calls ( having been caught in the act several times ) ,'
        ' and is generally the most athletic and nosiest of the Tanner girls . Her best friends in school are'
        ' Gia Mahan and Mickey , whom she meets in season seven ( the former is the only one who appears '
        'through to season eight ) . Of the three sisters , Stephanie has dealt with the toughest issues , '
        "such as peer pressure into smoking ( in season seven 's `` Fast Friends '' ) , `` make - out '' "
        "parties ( in season eight 's `` Making Out is Hard to Do '' ) , joyriding ( in season eight 's `` "
        "Stephanie 's Wild Ride '' ) , and uncovering a classmate 's child abuse ( in season six 's `` "
        "Silence is Not Golden '' ) , as well as the death of her mother when she was only five . In her "
        'early years , she is very sentimental about Mr. Bear , a stuffed animal that her mother gave to her '
        "after Michelle was born ( this was the focal point of the season two episode `` Goodbye Mr. Bear '' "
        ') . She and Jesse are the most abrasive when it comes to how they feel about Kimmy Gibbler .'),
    uid=4484,
    golds=('Michelangelo',),
)

ORDINARY = Example(
    question='who came up with the first working periodic table',
    passage=('The history of the periodic table reflects over a century of growth in the understanding of chemical'
        ' properties . The most important event in its history occurred in 1869 , when the table was '
        'published by Dmitri Mendeleev , who built upon earlier discoveries by scientists such as Antoine - '
        'Laurent de Lavoisier and John Newlands , but who is nevertheless generally given sole credit for its'
        ' development .'),
    partner_passage=('The essential nature of the atomic nucleus was established with the discovery of the neutron by '
        'James Chadwick in 1932 and the determination that it was a new elementary particle , distinct from '
        'the proton .'),
    uid=3177,
    golds=('Dmitri Mendeleev',),
)

#: (name, example, the answer the model has memorized)
EXAMPLES: Tuple[Tuple[str, Example, str], ...] = (
    ("NQ-Swap, a counterfactual passage", COUNTERFACTUAL, "James Brown"),
    ("NQ-Original, an ordinary passage", ORDINARY, "Dmitri Mendeleev"),
)

#: answers of Llama-3.1-8B, seed 0, for each method of Table 1
ANSWERS: Dict[str, Tuple[str, str]] = {
    "sampling": ("James Brown.", "Dmitri Mendeleev."),
    "cad": ("Michelangelo", "No one"),
    "adacad": ("James Brown.", "Dmitri Mendeleev"),
    "rpcd": ("Michelangelo Brown.", "Antoine-L Laurent de Lavoisier."),
    "id": ("Michelangelo", "No mention."),
    "fcd": ("Michelangelo", "Dmitri Mendeleev"),
}


def _short(text: str, words: int = 18) -> str:
    w = text.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def show_views(example: Example, words: int = 18) -> List[str]:
    """The full prompt and its two matched views, with long passages shortened for printing."""
    out = []
    names = {"full": "Full prompt", "directive_view": "Directive view (directive replaced)",
             "evidence_view": "Evidence view (passage replaced)"}
    for kind, text in views(example).items():
        instr, passage, question = text.split("\n\n", 2)
        label, body = passage.split(": ", 1)
        out += [f"[{names[kind]}]", f"  {instr}", f"  {label}: {_short(body, words)}", f"  {question}"]
    return out


def answers_table() -> List[str]:
    """The recorded answers of every method on the two questions, with their F1 against the passage answer."""
    rows = [f"{'Method':22s} {'counterfactual passage':34s} ordinary passage"]
    for m in TABLE1:
        cells = []
        for (_, ex, _), reply in zip(EXAMPLES, ANSWERS[m]):
            cells.append(f"{reply} (F1 {score(reply, ex):.2f})")
        rows.append(f"{LABELS[m]:22s} {cells[0]:34s} {cells[1]}")
    return rows


def main() -> None:
    name, ex, memory = EXAMPLES[0]
    print(f"{name}: {ex.question!r}, passage answer {ex.golds[0]!r}, memorized answer {memory!r}\n")
    print("\n".join(show_views(ex)))
    print("\nAnswers of Llama-3.1-8B (seed 0) on the two questions of Figure 1")
    print("\n".join(answers_table()))
