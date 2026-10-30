"""Generate small, deterministic synthetic inputs and CPU reference outputs."""
import argparse
import math
from pathlib import Path

import torch


def _expected(q, k, v, q_lengths=None, kv_lengths=None, key_padding_mask=None, *, causal, scale=None):
    batch, _, query_tokens, head_dim = q.shape
    key_tokens = k.shape[2]
    q_lengths = q_lengths if q_lengths is not None else torch.full((batch,), query_tokens)
    kv_lengths = kv_lengths if kv_lengths is not None else torch.full((batch,), key_tokens)
    key_padding_mask = key_padding_mask if key_padding_mask is not None else torch.ones((batch, key_tokens), dtype=torch.bool)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (scale if scale is not None else 1.0 / math.sqrt(head_dim))
    result = torch.zeros(q.shape, dtype=q.dtype)
    for b in range(batch):
        for row in range(int(q_lengths[b])):
            valid = (torch.arange(key_tokens) < kv_lengths[b]) & key_padding_mask[b]
            if causal:
                valid &= torch.arange(key_tokens) <= row + kv_lengths[b] - q_lengths[b]
            if bool(valid.any()):
                weights = torch.softmax(scores[b, :, row].masked_fill(~valid, -torch.inf), dim=-1)
                result[b, :, row] = (weights.unsqueeze(1) @ v[b].float()).squeeze(1).to(q.dtype)
    return result


def build_payload() -> dict:
    generator = torch.Generator(device="cpu").manual_seed(314159)
    prefill_q = torch.randn((1, 2, 7, 32), generator=generator, dtype=torch.float16)
    prefill_k = torch.randn((1, 2, 7, 32), generator=generator, dtype=torch.float16)
    prefill_v = torch.randn((1, 2, 7, 32), generator=generator, dtype=torch.float16)
    ragged_q = torch.randn((2, 3, 5, 16), generator=generator, dtype=torch.float16)
    ragged_k = torch.randn((2, 3, 9, 16), generator=generator, dtype=torch.float16)
    ragged_v = torch.randn((2, 3, 9, 16), generator=generator, dtype=torch.float16)
    ragged = {
        "q": ragged_q,
        "k": ragged_k,
        "v": ragged_v,
        "q_lengths": torch.tensor([5, 3], dtype=torch.int32),
        "kv_lengths": torch.tensor([9, 6], dtype=torch.int32),
        "key_padding_mask": torch.tensor([
            [True, True, False, True, True, False, True, True, True],
            [False, False, False, False, False, False, False, False, False],
        ]),
    }
    payload = {
        "prefill": {
            "q": prefill_q,
            "k": prefill_k,
            "v": prefill_v,
            "expected": _expected(prefill_q, prefill_k, prefill_v, causal=True),
        },
        "ragged-padding": {
            **ragged,
            "expected": _expected(ragged_q, ragged_k, ragged_v, ragged["q_lengths"], ragged["kv_lengths"],
                                  ragged["key_padding_mask"], causal=False, scale=0.25),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fixtures/demo-input.pt"))
    target = parser.parse_args().output
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_payload(), target)


if __name__ == "__main__":
    main()
