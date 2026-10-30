# Validation CLI format

The input `.pt` file is loaded with `weights_only=True` and contains a dictionary
of case IDs. Each case supplies `q`, `k`, `v`, and `expected` tensors. Optional
fields are `q_lengths`, `kv_lengths`, and `key_padding_mask`. The expected tensor
must have the same shape as Q. Only load inputs from trusted sources.

The config has exactly `schema_version` (1), `atol`, `rtol`, and `cases`.
Each case specifies a unique nonempty `id`, a boolean `causal`, and optionally
a finite positive `scale`. Tolerances must be finite and nonnegative.
See [the demo config](../configs/demo.json).

The output `.pt` dictionary maps each configured case ID to its computed tensor.
The JSON report contains exactly:

```text
schema_version, status, backend, output_path,
tested_configurations, summary
```

Each configuration record contains:

```text
id, shape, dtype, causal, valid_query_rows, fully_masked_rows,
max_abs_error, max_rel_error, passed
```

`shape` is `[batch, heads, queries, keys, head_dim]`. Row counts are per
`(batch, head, query)` triple. `valid_query_rows` excludes padded queries;
`fully_masked_rows` counts valid queries with no usable key.

The summary contains:

```text
total, passed, failed, max_abs_error, max_rel_error, atol, rtol
```

Comparison uses `abs(actual - expected) <= atol + rtol * abs(expected)`.
Relative error divides the absolute error by `max(abs(expected), 1e-12)`.
The backend is `triton` or `triton-interpreter`. Report keys are sorted, indentation
is two spaces, and a final newline is written. Non-finite JSON values are rejected.
Report status is `pass` only when every comparison passes.

Caught invalid-input or runtime errors print an explanation to stderr and exit
with code 2. A completed comparison run returns code 0 even when report status
is `fail`; consumers must check the report status rather than exit status alone.
This development snapshot does not guarantee atomic cleanup of output files
after every possible serialization or device failure.
