"""Decoders and FCD on a Hugging Face Transformers model.

    from fcd import Example
    from fcd.hf import FCD

    fcd = FCD.from_pretrained("Qwen/Qwen3-8B")          # gamma_D = 1.0, gamma_E = 0.5
    ex = Example(question="...", passage="...", partner_passage="... the passage of another example ...")
    print(fcd(ex, seed=0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .decoding import Generation, generate, make_rule
from .example import Example
from .methods import GAMMA_D, GAMMA_E, Method, fcd_method, get_method
from .models import REVISIONS, Sampling, resolve_model
from .prompts import build_prompt

#: maximum new tokens for short free-form and for multiple-choice answers
MAX_NEW_FREE, MAX_NEW_MC = 48, 32


@dataclass
class Output:
    texts: List[str]
    forward_tokens: int
    n_generated: List[int]


class Decoder:
    """A model, its tokenizer and its sampling settings, with every decoder of `fcd.methods.METHODS`."""

    def __init__(self, model, tokenizer, sampling: Sampling, *, max_batch: int = 64, kv_fraction: float = 0.55):
        self.model, self.tokenizer, self.sampling = model, tokenizer, sampling
        self.max_batch, self.kv_fraction = max_batch, kv_fraction
        self._thinking_off = "qwen3" in (getattr(tokenizer, "name_or_path", "") or "").lower()
        self._encoded: Dict[Tuple[Example, str], List[int]] = {}
        self.stop_ids = self._stop_ids()
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, name: str, *, device: Union[str, int] = 0, sampling: Optional[Sampling] = None,
                        revision: Optional[str] = None, attn_implementation: str = "sdpa", **kwargs) -> "Decoder":
        """Load a model in bfloat16. ``revision`` defaults to the revision of the paper's runs for its six models,
        and to the latest one for any other model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id, default = resolve_model(name)
        revision = revision or REVISIONS.get(hf_id)
        tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, dtype=torch.bfloat16,
                                                     device_map={"": device},
                                                     attn_implementation=attn_implementation).eval()
        return cls(model, tok, sampling or default, **kwargs)

    # ------------------------------------------------------------------ prompts

    def render(self, text: str) -> str:
        kw = {"enable_thinking": False} if self._thinking_off else {}
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, **kw)

    def encode(self, example: Example, kind: str) -> List[int]:
        key = (example, kind)
        if key not in self._encoded:
            text = self.render(build_prompt(example, kind))
            self._encoded[key] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self._encoded[key]

    def clear_cache(self) -> None:
        self._encoded.clear()

    def _stop_ids(self) -> List[int]:
        ids = set()
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif eos:
            ids.update(int(e) for e in eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return sorted(ids)

    # ------------------------------------------------------------------ decoding

    def generate(self, examples: Sequence[Example], method: Union[str, Method], *, seed: int = 0,
                 max_new_tokens: Optional[int] = None, log=None) -> Output:
        """Decode the examples with one method, a name of `fcd.methods.METHODS` or a `Method`."""
        m = get_method(method) if isinstance(method, str) else method
        if not examples:
            return Output([], 0, [])
        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_MC if all(ex.is_multiple_choice for ex in examples) else MAX_NEW_FREE
        streams = [[self.encode(ex, kind) for ex in examples] for kind in m.streams]
        gen: Generation = generate(self.model, streams, [ex.uid for ex in examples], rule=make_rule(m),
                                   sampling=self.sampling, seed=seed, max_new_tokens=max_new_tokens,
                                   stop_ids=self.stop_ids, pad_id=self.pad_id, max_batch=self.max_batch,
                                   kv_fraction=self.kv_fraction, log=log)
        texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in gen.tokens]
        return Output(texts, gen.forward_tokens, [len(t) for t in gen.tokens])


class FCD:
    """Factorized contrastive decoding over a `Decoder`.

    Every token is scored as l + gamma_d (l - l_D) + gamma_e (l - l_E), with l the log-probability under the full prompt,
    l_D under the directive view and l_E under the evidence view, and sampled with the model's sampling settings. The
    evidence view needs ``Example.partner_passage``, the passage of another example; the task loaders of `fcd.data`
    fill it in.
    """

    def __init__(self, decoder: Decoder, gamma_d: float = GAMMA_D, gamma_e: float = GAMMA_E):
        self.decoder = decoder
        self.method = fcd_method(gamma_d, gamma_e)

    @classmethod
    def from_pretrained(cls, name: str, *, gamma_d: float = GAMMA_D, gamma_e: float = GAMMA_E,
                        device: Union[str, int] = 0, sampling: Optional[Sampling] = None,
                        revision: Optional[str] = None, **kwargs) -> "FCD":
        return cls(Decoder.from_pretrained(name, device=device, sampling=sampling, revision=revision, **kwargs),
                   gamma_d, gamma_e)

    def generate(self, examples: Sequence[Example], seed: int = 0, log=None) -> Output:
        return self.decoder.generate(examples, self.method, seed=seed, log=log)

    def __call__(self, examples: Union[Example, Sequence[Example]], seed: int = 0):
        if isinstance(examples, Example):
            return self.generate([examples], seed).texts[0]
        return self.generate(examples, seed).texts
