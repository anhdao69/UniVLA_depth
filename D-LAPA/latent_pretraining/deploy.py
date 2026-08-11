# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from latent_pretraining.sampler_latent_action_pretrain import DeltaActionSampler
from latent_pretraining.sampler_action_pretrain import ActionSampler
from tux import JaxDistributedConfig
from latent_pretraining.delta_llama import VideoLLaMAConfig
from tux import JaxDistributedConfig, set_random_seed
import csv

class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


# === Server Interface ===
class LAPAServer:
    def __init__(
            self, 
            load_checkpoint: Union[str, Path], 
            vqgan_checkpoint: Union[str, Path], 
            seed: int,
            mesh_dim: str, 
            dtype: str, 
            load_llama_config: str, 
            updata_llama_config: str, 
            tokens_per_delta: int, 
            tokens_per_action: int, 
            vocab_file: str, 
            multi_image: int, 
            jax_distributed: dict,
            action_scale_file: str,
            img_aug: int,
            stage25_feature_server_url: str = "",
        ) -> Path:
        
        set_random_seed(seed)
        tokenizer = VideoLLaMAConfig.get_tokenizer_config()
        llama = VideoLLaMAConfig.get_default_config()
        tokenizer.vocab_file = vocab_file

        # Read the bin-edge table first: the per-dim bin counts are handed to the
        # sampler so generation can mask out-of-range action logits (each dim has
        # len(edges)-1 valid bins, usually fewer than action_vocab_size).
        self.action_scale_list = []
        with open(action_scale_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                # Convert the string values to float and add them to the csv_data list
                self.action_scale_list.append([float(value) for value in row if value.strip()])
        action_dim_sizes = [len(edges) - 1 for edges in self.action_scale_list]

        kwargs = {
            "vqgan_checkpoint": vqgan_checkpoint,
            "seed": seed,
            "mesh_dim": mesh_dim,
            "dtype": dtype,
            "load_llama_config": load_llama_config,
            "update_llama_config": updata_llama_config,
            "tokens_per_delta": tokens_per_delta,
            "tokens_per_action": tokens_per_action,
            "vocab_file": vocab_file,
            "multi_image": multi_image,
            "jax_distributed": jax_distributed,
            "action_scale_file": action_scale_file,
            "action_dim_sizes": action_dim_sizes,
            "tokenizer": tokenizer,
            "llama": llama,
            "load_checkpoint": load_checkpoint,
            "image_aug": img_aug,
        }
        self.tokens_per_delta = tokens_per_delta
        self.cnt = 0
        flags = FLAGSClass(kwargs)

        if kwargs['tokens_per_delta'] > 0:
            self.model = DeltaActionSampler(flags)
        else:
            self.model = ActionSampler(flags)
        self.load_checkpoint= load_checkpoint
        self.stage25_feature_server_url = stage25_feature_server_url.rstrip("/")

    def predict_action(self, payload: Dict[str, Any]) -> str:
        self.cnt +=1
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            image_path, instruction = payload["image"], payload["instruction"]
            print(f'image_path: {image_path}')
            image = Image.open(image_path)
            # Convert the image to a NumPy array
            image = np.array(image)

            prompt = {'image': [image], 'question': instruction}
            depth_feature = payload.get("depth_feature")
            stage25_debug = None
            if depth_feature is not None:
                prompt["depth_feature"] = np.asarray(depth_feature, dtype=np.float32)
            elif self.stage25_feature_server_url:
                stage25_request = {
                    "image": image_path,
                    "instruction": instruction,
                    "return_debug": payload.get("return_debug", False),
                }
                if payload.get("depth_image") is not None:
                    stage25_request["depth_image"] = payload["depth_image"]
                response = requests.post(
                    f"{self.stage25_feature_server_url}/feature",
                    json=stage25_request,
                    timeout=180,
                )
                response.raise_for_status()
                stage25_payload = response.json()
                if "error" in stage25_payload:
                    raise RuntimeError(stage25_payload["error"])
                prompt["depth_feature"] = np.asarray(
                    stage25_payload["z_depth_feature_pred"], dtype=np.float32
                )
                stage25_debug = {
                    "model_name": stage25_payload.get("model_name"),
                    "z_depth_shape": stage25_payload.get("z_depth_shape"),
                    "z_rgb_shape": stage25_payload.get("z_rgb_shape"),
                    "depth_source": stage25_payload.get("depth_source"),
                }

            prompts = [prompt]

            action_outputs = self.model(prompts)
            action_outputs = action_outputs[0]


            action = self.get_averaged_values(action_outputs)
            if payload.get("return_debug", False):
                action = {
                    "action": action,
                    "action_tokens": np.asarray(action_outputs).tolist(),
                }
                if stage25_debug is not None:
                    action["stage25"] = stage25_debug

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"
    def get_averaged_values(self, indices):
        averaged_values = []
        for row_idx, idx in enumerate(indices):
            # The action head has action_vocab_size classes for every dim, but each
            # dim's bin table can be shorter (qcut drops duplicate edges; gripper has
            # only 2 bins). A predicted index outside this dim's bins is clamped to
            # the nearest valid bin instead of silently becoming a bogus action.
            edges = self.action_scale_list[row_idx]
            n_bins = len(edges) - 1
            clamped = min(max(int(idx), 0), n_bins - 1)
            if clamped != idx:
                print(f"action dim {row_idx}: predicted bin {idx} outside [0, {n_bins - 1}], clamped to {clamped}")
            average = (edges[clamped] + edges[clamped + 1]) / 2
            averaged_values.append(average)
        return averaged_values

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

@dataclass
class DeployConfig:
    # fmt: off
    load_checkpoint: Union[str, Path] = None

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 32820                                                    # Host Port
    vqgan_checkpoint: str = "lapa_checkpoints/vqgan"
    seed: int = 1234
    mesh_dim: str = "1,-1,1,1"
    dtype: str = "bf16"
    load_llama_config: str = "7b"
    update_llama_config: str = "dict(action_vocab_size=256,delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" 
    tokens_per_delta: int = 4
    tokens_per_action: int = 7
    vocab_file: str = "lapa_checkpoints/tokenizer.model"
    multi_image: int = 1
    image_aug: int = 1
    jax_distributed: dict = JaxDistributedConfig.get_default_config()
    action_scale_file: str = None
    stage25_feature_server_url: str = ""


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = LAPAServer(
        cfg.load_checkpoint,
        cfg.vqgan_checkpoint,
        cfg.seed,
        cfg.mesh_dim,
        cfg.dtype,
        cfg.load_llama_config,
        cfg.update_llama_config,
        cfg.tokens_per_delta,
        cfg.tokens_per_action,
        cfg.vocab_file,
        cfg.multi_image,
        cfg.jax_distributed,
        cfg.action_scale_file,
        cfg.image_aug,
        cfg.stage25_feature_server_url,
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
