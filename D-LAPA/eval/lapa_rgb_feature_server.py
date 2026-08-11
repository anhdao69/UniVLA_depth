#!/usr/bin/env python3
"""HTTP server for baseline LAPA RGB feature extraction.

This runs the original LAPA model in its own process/GPU and exposes the
4096-D RGB feature needed by the Stage-2.5 depth branch.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image


def import_stage25_bundle(bundle_dir: str):
    bundle = Path(bundle_dir).resolve()
    laq_dir = bundle / "laq"
    if not (laq_dir / "rollout_stage25_model4.py").exists():
        raise FileNotFoundError(f"Cannot find laq/rollout_stage25_model4.py under {bundle}")
    for path in (str(bundle), str(laq_dir)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    from rollout_stage25_model4 import build_lapa  # noqa: E402

    return build_lapa


def load_rgb(path: str, image_size: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((image_size, image_size)), dtype=np.uint8)


class LAPARgbFeatureServer:
    def __init__(self, args):
        build_lapa = import_stage25_bundle(args.stage25_bundle_dir)
        self.image_size = args.image_size
        print(
            json.dumps(
                {
                    "lapa_rgb_feature_server": {
                        "bundle": str(Path(args.stage25_bundle_dir).resolve()),
                        "checkpoint": args.original_lapa_checkpoint,
                        "mesh_dim": args.mesh_dim,
                    }
                }
            ),
            flush=True,
        )
        self.lapa = build_lapa(
            tokens_per_delta=args.tokens_per_delta,
            vqgan_checkpoint=args.vqgan_checkpoint,
            vocab_file=args.vocab_file,
            load_checkpoint=args.original_lapa_checkpoint,
            image_size=args.image_size,
            multi_image=args.multi_image,
            seed=args.seed,
            mesh_dim=args.mesh_dim,
            dtype=args.dtype,
            load_llama_config=args.load_llama_config,
        )

    def feature(self, payload: Dict[str, Any]):
        try:
            rgb = load_rgb(payload["image"], self.image_size)
            instruction = payload["instruction"]
            latent_action, z_rgb_feature = self.lapa.inference(
                rgb,
                instruction,
                return_feature=True,
                feature_pool=payload.get("feature_pool", "mean"),
            )
            z_rgb = np.asarray(z_rgb_feature, dtype=np.float32)
            result = {
                "z_rgb_feature": z_rgb.tolist(),
                "z_rgb_shape": list(z_rgb.shape),
            }
            if payload.get("return_debug", False):
                result["latent_action"] = np.asarray(latent_action).tolist()
            return JSONResponse(result)
        except Exception:
            traceback.print_exc()
            return JSONResponse({"error": traceback.format_exc()}, status_code=500)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline LAPA RGB feature server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32822)
    parser.add_argument("--stage25_bundle_dir", type=str, required=True)
    parser.add_argument("--original_lapa_checkpoint", type=str, required=True)
    parser.add_argument("--vqgan_checkpoint", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--tokens_per_delta", type=int, default=4)
    parser.add_argument("--multi_image", type=int, default=1)
    parser.add_argument("--mesh_dim", type=str, default="1,1,1,1")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--load_llama_config", type=str, default="7b")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    server = LAPARgbFeatureServer(args)
    app = FastAPI()
    app.post("/rgb_feature")(server.feature)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
