"""Sampling noise, filtering, decoding rules and the generation loop. Needs torch (and transformers for the last test)."""

import math

import pytest

torch = pytest.importorskip("torch")

from fcd.decoding import (filter_scores, generate, gumbel_noise, make_rule, step_keys,  # noqa: E402
                          uniform_noise)
from fcd.methods import fcd_method, get_method  # noqa: E402
from fcd.models import Sampling  # noqa: E402


def test_noise_is_a_fixed_function_of_seed_uid_step_and_token():
    uids = torch.tensor([0, 308, 912, 3674, 948])
    assert step_keys(0, uids, 0).tolist() == [3956773149, 156408894, 951316562, 2192675301, 3366838693]
    assert step_keys(2, uids, 5).tolist() == [101931727, 3848188342, 1987834672, 626881271, 1448070327]
    u = uniform_noise(step_keys(0, torch.tensor([308]), 0), 6)
    assert ((u * 8388608.0) - 0.5).round().long().tolist() == [[3742466, 3448000, 5404781, 7504643, 2492207, 6099218]]


def test_noise_range():
    u = uniform_noise(step_keys(1, torch.arange(64), 3), 4096)
    assert float(u.min()) > 0.0 and float(u.max()) < 1.0
    assert torch.isfinite(gumbel_noise(step_keys(1, torch.arange(64), 3), 4096)).all()
    # a row's noise does not depend on the other rows
    assert torch.equal(uniform_noise(step_keys(1, torch.tensor([5]), 3), 100)[0],
                       uniform_noise(step_keys(1, torch.tensor([9, 5]), 3), 100)[1])


def test_filter_top_k_then_top_p():
    x = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.06, 0.04]]))
    kept = torch.isfinite(filter_scores(x, Sampling(1.0, 0.8, 3))).tolist()[0]
    assert kept == [True, True, False, False, False]          # top 3, then 0.5 + 0.3 reaches 0.8
    assert torch.isfinite(filter_scores(x, Sampling(1.0, 1.0, 0))).all()


def lp(*rows):
    return torch.log(torch.tensor(rows, dtype=torch.float32)).unsqueeze(1)    # [streams, 1, vocab]


def test_guidance_rules():
    p = lp([0.6, 0.3, 0.1], [0.2, 0.7, 0.1])
    assert torch.allclose(make_rule(get_method("cad"))(p), 2 * p[0] - p[1])
    assert torch.allclose(make_rule(get_method("rpcd"))(p), 2 * p[0] - p[1])
    assert torch.allclose(make_rule(get_method("id"))(p), 3 * p[0] - 2 * p[1])
    assert torch.equal(make_rule(get_method("sampling"))(p[:1]), p[0])


def test_fcd_rule_and_its_special_cases():
    p = lp([0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.1, 0.4])       # full prompt, directive view, evidence view
    got = make_rule(get_method("fcd"))(p)
    assert torch.allclose(got, p[0] + 1.0 * (p[0] - p[1]) + 0.5 * (p[0] - p[2]))
    assert torch.allclose(make_rule(fcd_method(0.0, 0.0))(p), p[0])
    assert torch.allclose(make_rule(fcd_method(2.0, 0.0))(p), make_rule(get_method("id"))(p[:2]))
    assert torch.allclose(make_rule(fcd_method(0.0, 1.0))(p), make_rule(get_method("rpcd"))(p[[0, 2]]))
    assert get_method("fcd").streams == ("full", "directive_view", "evidence_view")


def test_adacad_scales_by_js_divergence():
    same = lp([0.6, 0.3, 0.1], [0.6, 0.3, 0.1])
    assert torch.allclose(make_rule(get_method("adacad"))(same), same[0])
    far = lp([0.98, 0.01, 0.01], [0.01, 0.98, 0.01])
    m = 0.5 * (far[0].exp() + far[1].exp())
    js = 0.5 * (far[0].exp() * (far[0] - m.log())).sum() + 0.5 * (far[1].exp() * (far[1] - m.log())).sum()
    expected = far[0] + js / math.log(2) * (far[0] - far[1])
    assert torch.allclose(make_rule(get_method("adacad"))(far), expected, atol=1e-5)


def tiny_model():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_generate_masks_counts_and_repeats():
    model = tiny_model()
    prompts = [[1, 5, 9, 13], [1, 7, 11], [1, 2, 3, 4, 5, 6]]
    kw = dict(rule=make_rule(get_method("sampling")), sampling=Sampling(1.0, 1.0, 0), seed=0, max_new_tokens=6,
              stop_ids=[], pad_id=0)
    a = generate(model, [prompts], [10, 11, 12], **kw)
    b = generate(model, [prompts], [10, 11, 12], **kw)
    assert a.tokens == b.tokens and all(len(t) == 6 for t in a.tokens)
    # prompts once, then one token per row at each of the 5 later steps
    assert a.forward_tokens == sum(map(len, prompts)) + 5 * len(prompts)
    banned = sorted({t for row in a.tokens for t in row})
    c = generate(model, [prompts], [10, 11, 12], masks=[banned, [], banned], **kw)
    assert not set(c.tokens[0]) & set(banned) and not set(c.tokens[2]) & set(banned)
    assert c.tokens[1] == a.tokens[1]


def test_two_stream_rule_with_zero_strength_matches_plain_sampling():
    model = tiny_model()
    prompts, contrast = [[1, 5, 9, 13]], [[1, 13]]
    kw = dict(sampling=Sampling(0.7, 0.8, 20), seed=3, max_new_tokens=8, stop_ids=[], pad_id=0, compact=False)
    plain = generate(model, [prompts], [7], rule=make_rule(get_method("sampling")), **kw)
    cad0 = generate(model, [prompts, contrast], [7], rule=make_rule(get_method("cad", [0.0])), **kw)
    fcd0 = generate(model, [prompts, contrast, contrast], [7], rule=make_rule(fcd_method(0.0, 0.0)), **kw)
    assert plain.tokens == cad0.tokens == fcd0.tokens


def test_three_streams_count_every_prompt_and_step():
    model = tiny_model()
    full, dview, eview = [[1, 5, 9, 13]], [[1, 6, 9, 13]], [[1, 5, 10, 11, 13]]
    kw = dict(sampling=Sampling(1.0, 1.0, 0), seed=1, max_new_tokens=5, stop_ids=[], pad_id=0, compact=False)
    out = generate(model, [full, dview, eview], [3], rule=make_rule(get_method("fcd")), **kw)
    # the three prompts, padded to the longest, then one token per stream at each of the 4 later steps
    assert len(out.tokens[0]) == 5
    assert out.forward_tokens == 4 + 4 + 5 + 3 * 4


def test_dropping_finished_rows_keeps_every_answer():
    model = tiny_model()
    n = 32
    prompts = [[1, 3 + i % 7, 5 + i % 11, 9 + i % 5] for i in range(n)]
    contrast = [[1, 9 + i % 5] for i in range(n)]
    kw = dict(rule=make_rule(get_method("cad")), sampling=Sampling(1.0, 1.0, 0), seed=2, max_new_tokens=64,
              stop_ids=[60, 61, 62, 63], pad_id=0)
    kept = generate(model, [prompts, contrast], list(range(n)), compact=False, **kw)
    dropped = generate(model, [prompts, contrast], list(range(n)), compact=True, **kw)
    assert dropped.tokens == kept.tokens
    assert dropped.forward_tokens < kept.forward_tokens      # finished rows left the batch and the cache

