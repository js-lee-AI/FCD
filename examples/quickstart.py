"""FCD on the two test questions of the paper's Figure 1.

    python examples/quickstart.py                         # the views and the recorded answers (CPU, no model)
    python examples/quickstart.py --model llama3.1-8b     # decodes the same questions with a model (GPU)

The views of the counterfactual question change one factor each: the directive view replaces the directive by its
opposite, and the evidence view replaces the passage by that of another example. FCD contrasts the full prompt with
both, with the strengths gamma_D = 1.0 and gamma_E = 0.5.
"""

import argparse

from fcd import TABLE1
from fcd.demo import EXAMPLES, answers_table, show_views
from fcd.methods import LABELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a short name such as llama3.1-8b, or a Hugging Face id; omit to use the "
                                    "recorded answers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name, ex, memory = EXAMPLES[0]
    print(f"{name}: {ex.question!r}")
    print(f"  the passage says {ex.golds[0]!r}, the model has memorized {memory!r}\n")
    print("\n".join(show_views(ex)))

    if not args.model:
        print("\nAnswers of Llama-3.1-8B (seed 0), as recorded in the paper's runs")
        print("\n".join(answers_table()))
        return

    from fcd import score
    from fcd.hf import Decoder

    dec = Decoder.from_pretrained(args.model)
    examples = [e for _, e, _ in EXAMPLES]
    print(f"\nAnswers of {args.model} (seed {args.seed})")
    for m in TABLE1:
        replies = dec.generate(examples, m, seed=args.seed).texts
        cells = [f"{r!r} (F1 {score(r, e):.2f})" for r, e in zip(replies, examples)]
        print(f"  {LABELS[m]:22s} {cells[0]:36s} {cells[1]}")


if __name__ == "__main__":
    main()
