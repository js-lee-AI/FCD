"""fcd: factorized contrastive decoding, training-free decoding that follows the passage in the prompt.

    from fcd import Example, views

    ex = Example(question="who wrote papa got a brand new bag",
                 passage="... is a song written and recorded by Michelangelo ...",
                 partner_passage="... the passage of another example ...")
    views(ex)                                   # the full prompt, the directive view and the evidence view

    from fcd.hf import FCD                      # needs torch and transformers
    fcd = FCD.from_pretrained("qwen3-8b")       # gamma_D = 1.0, gamma_E = 0.5
    fcd(ex, seed=0)                             # the answer

The prompts, the methods and the scoring need only the standard library. Decoding needs torch and transformers
(fcd.hf, fcd.decoding), the task loaders need datasets (fcd.data), and the tables need numpy (fcd.stats, fcd.table);
these modules are imported on first use.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .example import Example
from .methods import GAMMA_D, GAMMA_E, METHODS, TABLE1, Method, fcd_method, get_method
from .prompts import build_prompt, views
from .scoring import f1_score, parse_letter, score

# names that need torch and transformers, imported on first attribute access (PEP 562)
_LAZY = {"FCD": "hf", "Decoder": "hf"}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "Example", "build_prompt", "views",
    "Method", "METHODS", "TABLE1", "GAMMA_D", "GAMMA_E", "fcd_method", "get_method",
    "score", "f1_score", "parse_letter",
    "FCD", "Decoder",
]
