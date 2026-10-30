# Triton Fused Attention

A correctness-focused Transformer attention implementation in Python and Triton,
with a PyTorch reference, reproducible synthetic inputs, and a JSON validation CLI.
This is a standalone development snapshot extracted from my attention-kernel
repair project, not a complete inference engine or a performance claim.

## What is implemented

- Q shaped `[batch, heads, queries, head_dim]`, with K and V shaped
  `[batch, heads, keys, head_dim]`.
- Head dimensions 16, 32, 64, and 128; float16, bfloat16, and float32 inputs.
- Bottom-right causal alignment, per-batch query/key prefix lengths, and a
  boolean key-validity mask (`True` means usable).
- Stride-aware input addressing, including non-contiguous views; contiguous
  output with the same shape and dtype as Q.
- Float32, two-pass tiled softmax; normalized weights are multiplied by V
  without first forming a potentially overflowing unnormalized value sum.
- Zero output for padded query rows and fully masked rows, plus input validation.
- A CLI that compares outputs with supplied reference tensors and writes a
  deterministic, strict JSON report.

Physical batch, head, query, and key dimensions must be positive. Valid prefix
lengths may be zero. All input tensors must share their device and floating
dtype; lengths use int32/int64 and the key mask uses bool.

## Setup

Kernel execution targets **Linux with an NVIDIA CUDA GPU**, using Python
3.10-3.12, PyTorch 2.5.1, Triton 3.1.0, and NumPy 1.26.4. Install an appropriate
NVIDIA driver first. Native Windows does not execute this Triton kernel.
No Docker is required.

From a clone, on Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The generator and reference-only tests need PyTorch but not Triton, and can run
on CPU, including Windows. A supported Linux Triton installation can also use
`TRITON_INTERPRET=1` for CPU diagnostics; set this before starting Python.
Interpreter diagnostics do not establish GPU correctness or performance.

## Try the API

```python
import torch
from fused_attention import fused_attention

q = torch.randn(1, 2, 7, 32, device="cuda", dtype=torch.float16)
k = torch.randn(1, 2, 9, 32, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
output = fused_attention(q, k, v, causal=True)
print(output.shape)  # torch.Size([1, 2, 7, 32])
```

For query index `i`, bottom-right causal masking permits a key index `j` when
`j <= i + kv_length - q_length`, in addition to the length and padding masks.
The default scale is `1 / sqrt(head_dim)`; an explicit scale must be finite
and positive.

## Generate inputs and validate

The demo includes prefill and ragged-padding cases with independently computed
CPU reference outputs. Generated tensor files and reports are intentionally
not committed.

```bash
python tools/generate_fixture.py --output fixtures/demo-input.pt
python -m fused_attention.cli \
  --input fixtures/demo-input.pt \
  --config configs/demo.json \
  --output results/attention-output.pt \
  --report results/kernel-report.json
```

See [the report format](docs/report-format.md). The CLI uses CUDA when available;
without CUDA it requires explicit Triton interpreter mode. Ordinary CPU execution
is rejected. A well-formed run can exit successfully even when numerical
comparisons fail: inspect `status` and `summary.failed` in the report.

## Developer tests

```bash
python -m unittest discover -s tests -v
```

Reference tests run on CPU. Kernel tests run only when CUDA and Triton are
available, or when Linux Triton interpreter mode is explicitly enabled; otherwise
they are reported as skipped. These are small development tests, not the original
benchmark acceptance suite. A reference-only pass does not mean the GPU kernel
has been tested on the current machine.

## Development status

Claude Code has been used to assist development of the original project. This
repository contains the current source snapshot, not reconstructed development
history or AI conversation logs. Code and technical claims still need human review.

The implementation prioritizes understandable numerical behavior over speed.
There are no speedup claims or published GPU measurements here. Further work
includes profiling on representative GPUs, broader randomized differential
testing, CUDA memory-safety checks, and evaluating faster tiling strategies without
changing masking or numerical semantics.

This repository deliberately excludes external evaluation harnesses, scoring
scripts, model rollout exports, task-upload archives, platform guidelines,
review receipts, and private work records. No open-source license is selected
in this snapshot; public visibility alone does not grant reuse rights.

This project was developed through an AI-assisted engineering workflow using Claude Code, with human direction over the architecture, debugging strategy, edge-case design, and verification. The work emphasized iterative diagnosis and executable regression tests rather than one-shot code generation.
