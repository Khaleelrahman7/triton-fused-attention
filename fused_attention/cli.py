from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import torch

from . import kernel


TOP_KEYS = {"schema_version", "status", "backend", "output_path", "tested_configurations", "summary"}


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if set(config) != {"schema_version", "atol", "rtol", "cases"}:
        raise ValueError("config fields are invalid")
    for name in ("atol", "rtol"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    cases = config["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty array")
    ids = set()
    for item in cases:
        if not isinstance(item, dict) or set(item) - {"id", "causal", "scale"} or set(item) < {"id", "causal"}:
            raise ValueError("case fields are invalid")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
            raise ValueError("case IDs must be unique nonempty strings")
        if type(item["causal"]) is not bool:
            raise ValueError("causal must be boolean")
        if "scale" in item and (isinstance(item["scale"], bool) or not isinstance(item["scale"], (int, float)) or not math.isfinite(item["scale"]) or item["scale"] <= 0):
            raise ValueError("scale must be finite and positive")
        ids.add(item["id"])
    return config


def _row_counts(item: dict, causal: bool) -> tuple[int, int]:
    q = item["q"]
    batch, heads, q_tokens, _ = q.shape
    key_tokens = item["k"].shape[2]
    q_lengths = item.get("q_lengths")
    kv_lengths = item.get("kv_lengths")
    key_mask = item.get("key_padding_mask")
    if q_lengths is None:
        q_lengths = torch.full((batch,), q_tokens, device=q.device, dtype=torch.long)
    if kv_lengths is None:
        kv_lengths = torch.full((batch,), key_tokens, device=q.device, dtype=torch.long)
    if key_mask is None:
        key_mask = torch.ones((batch, key_tokens), device=q.device, dtype=torch.bool)
    valid_queries = int(q_lengths.sum().item()) * heads
    empty = 0
    for b in range(batch):
        for row in range(int(q_lengths[b].item())):
            limit = row + int(kv_lengths[b].item()) - int(q_lengths[b].item()) if causal else key_tokens - 1
            usable = key_mask[b, : int(kv_lengths[b].item())]
            if causal:
                usable = usable[: max(0, min(int(kv_lengths[b].item()), limit + 1))]
            if not bool(usable.any().item()):
                empty += heads
    return valid_queries, empty


def run(input_path: Path, config_path: Path, output_path: Path, report_path: Path) -> None:
    config = _load_config(config_path)
    if torch.cuda.is_available():
        execution_device = "cuda"
    elif os.environ.get("TRITON_INTERPRET") == "1":
        execution_device = "cpu"
    else:
        raise RuntimeError("CUDA is unavailable")
    payload = torch.load(input_path, map_location=execution_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("input must be a dictionary")
    outputs = {}
    records = []
    atol, rtol = float(config["atol"]), float(config["rtol"])
    for case in config["cases"]:
        case_id = case["id"]
        if case_id not in payload or not isinstance(payload[case_id], dict):
            raise ValueError(f"missing input case: {case_id}")
        item = payload[case_id]
        if not {"q", "k", "v", "expected"} <= set(item):
            raise ValueError(f"case {case_id} lacks q, k, v, or expected")
        expected = item["expected"]
        if not isinstance(expected, torch.Tensor) or not isinstance(item["q"], torch.Tensor) or expected.shape != item["q"].shape:
            raise ValueError(f"case {case_id} expected must be a tensor shaped like q")
        kwargs = {name: item.get(name) for name in ("q_lengths", "kv_lengths", "key_padding_mask")}
        kwargs.update(causal=case["causal"], scale=case.get("scale"))
        actual = kernel.fused_attention(item["q"], item["k"], item["v"], **kwargs)
        outputs[case_id] = actual.detach().cpu()
        delta = (actual.float() - expected.float()).abs()
        finite = bool(torch.isfinite(actual).all().item())
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        relative = delta / expected.float().abs().clamp_min(1e-12)
        max_rel = float(relative.max().item()) if relative.numel() else 0.0
        passed = finite and bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
        valid_rows, empty_rows = _row_counts(item, case["causal"])
        b, h, qn, d = item["q"].shape
        kn = item["k"].shape[2]
        records.append({
            "id": case_id,
            "shape": [b, h, qn, kn, d],
            "dtype": str(item["q"].dtype).removeprefix("torch."),
            "causal": case["causal"],
            "valid_query_rows": valid_rows,
            "fully_masked_rows": empty_rows,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "passed": passed,
        })
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, output_path)
    passed_count = sum(record["passed"] for record in records)
    report = {
        "schema_version": 1,
        "status": "pass" if passed_count == len(records) else "fail",
        "backend": "triton" if execution_device == "cuda" else "triton-interpreter",
        "output_path": str(output_path),
        "tested_configurations": records,
        "summary": {
            "total": len(records),
            "passed": passed_count,
            "failed": len(records) - passed_count,
            "max_abs_error": max(record["max_abs_error"] for record in records),
            "max_rel_error": max(record["max_rel_error"] for record in records),
            "atol": atol,
            "rtol": rtol,
        },
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.input, args.config, args.output, args.report)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError, RuntimeError,
            pickle.UnpicklingError, EOFError) as exc:
        return _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
