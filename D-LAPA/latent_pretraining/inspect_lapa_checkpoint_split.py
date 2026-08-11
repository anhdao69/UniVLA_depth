import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import jax
import numpy as np
from flax.traverse_util import flatten_dict
from tux import StreamingCheckpointer


def _shape(value: Any) -> Tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    return tuple()


def _dtype(value: Any) -> str:
    if hasattr(value, "dtype"):
        return str(value.dtype)
    return type(value).__name__


def _numel(shape: Iterable[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _bytes_for(dtype: str, numel: int) -> int:
    try:
        return int(np.dtype(dtype).itemsize * numel)
    except TypeError:
        if "bfloat16" in dtype or "float16" in dtype:
            return 2 * numel
        if "float32" in dtype or "int32" in dtype:
            return 4 * numel
        return 0


def _group(path: str) -> str:
    if "depth_action_proj" in path:
        return "depth_action_proj"
    if "action_head" in path:
        return "action_head"
    if "transformer" in path:
        return "lapa_transformer"
    if "vision_head" in path:
        return "vision_head"
    if "lm_head" in path:
        return "lm_head"
    if "delta_head" in path:
        return "delta_head"
    return "other"


def _summarize(params: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    flat = flatten_dict(params, sep="/")
    groups: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tensors": 0, "params": 0, "bytes": 0})
    rows = []

    for path, value in flat.items():
        shape = _shape(value)
        dtype = _dtype(value)
        numel = _numel(shape)
        nbytes = _bytes_for(dtype, numel)
        group = _group(path)
        groups[group]["tensors"] += 1
        groups[group]["params"] += numel
        groups[group]["bytes"] += nbytes
        rows.append(
            {
                "path": path,
                "group": group,
                "shape": list(shape),
                "dtype": dtype,
                "params": numel,
                "bytes": nbytes,
            }
        )

    rows.sort(key=lambda x: x["params"], reverse=True)
    total_params = sum(row["params"] for row in rows)
    total_bytes = sum(row["bytes"] for row in rows)

    group_rows = []
    for name, stats in sorted(groups.items(), key=lambda item: item[1]["params"], reverse=True):
        group_rows.append(
            {
                "group": name,
                "tensors": stats["tensors"],
                "params": stats["params"],
                "bytes": stats["bytes"],
                "param_fraction": stats["params"] / total_params if total_params else 0.0,
            }
        )

    return {
        "total_tensors": len(rows),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "groups": group_rows,
        "largest_tensors": rows[:top_k],
        "head_tensors": [
            row
            for row in rows
            if row["group"] in {"depth_action_proj", "action_head"}
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a LAPA/LAPA-Depth streaming params checkpoint and estimate a trunk/head split."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint spec, e.g. params::/path/to/streaming_params")
    parser.add_argument("--output_json", default="", help="Optional path to write the summary JSON.")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--max_buffer_gb", type=float, default=float(os.environ.get("MAX_BUFFER_GB", "32")))
    args = parser.parse_args()

    max_buffer_size = int(args.max_buffer_gb * (2**30))
    print(json.dumps({"loading_checkpoint": args.checkpoint, "max_buffer_gb": args.max_buffer_gb}))
    _, params = StreamingCheckpointer.load_trainstate_checkpoint(
        args.checkpoint,
        disallow_trainstate=True,
        max_buffer_size=max_buffer_size,
    )

    if isinstance(params, dict) and set(params.keys()) == {"params"}:
        params_for_summary = params["params"]
    else:
        params_for_summary = params

    summary = _summarize(params_for_summary, args.top_k)
    summary["checkpoint"] = args.checkpoint
    summary["jax_platforms"] = os.environ.get("JAX_PLATFORMS", "")
    summary["jax_devices"] = [str(device) for device in jax.devices()]

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(json.dumps({"wrote": str(out)}))


if __name__ == "__main__":
    main()
