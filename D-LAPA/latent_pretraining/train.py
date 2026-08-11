import pprint
import os
import shutil
import subprocess
import copy

from tqdm import tqdm, trange
import numpy as np
from absl.app import run
import absl.logging as logging
import tux

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from latent_pretraining.data import DatasetFactory
from tux import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_mask,
    make_shard_and_gather_fns, with_sharding_constraint, define_flags_with_default,
    OptimizerFactory, StreamingCheckpointer
)
from latent_pretraining.llama import LLaMAConfig, FlaxLLaMAForCausalLMModule
from latent_pretraining.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLMModule
from latent_pretraining.delta_llama import VideoLLaMAConfig, FlaxDeltaLaMAForCausalLMModule
from latent_pretraining.llama_action import VideoLLaMAConfig, FlaxActionLaMAForCausalLMModule
from latent_pretraining.delta_llama_action import VideoLLaMAConfig, FlaxDeltaActionLaMAForCausalLMModule
import random

import flax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from flax.serialization import (from_bytes, from_state_dict, to_state_dict)
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict
from tux.utils import open_file
import tensorflow as tf
tf.config.optimizer.set_jit(True)
import time
try:
    import resource
except ImportError:
    resource = None

random.seed(time.time())


def _format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _process_rss_bytes():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB, macOS reports bytes. This repo runs training on Linux.
    return int(usage.ru_maxrss) * 1024


def _gpu_memory_summary():
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "nvidia-smi not found"
    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception as exc:
        return f"nvidia-smi failed: {type(exc).__name__}: {exc}"
    lines = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            idx, name, used, total = parts
            lines.append(f"gpu{idx} {name}: {used}/{total} MiB")
        elif line.strip():
            lines.append(line.strip())
    return "; ".join(lines) if lines else "no GPU rows"


def _shape_dtype(value):
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        return type(value).__name__
    return f"shape={tuple(shape)}, dtype={dtype}"


def _batch_summary(batch):
    if not isinstance(batch, dict):
        return _shape_dtype(batch)
    items = []
    for key in sorted(batch.keys()):
        items.append(f"{key}: {_shape_dtype(batch[key])}")
    return "{ " + "; ".join(items) + " }"


def _runtime_log(label, extra=None):
    if jax.process_index() != 0:
        return
    rss = _process_rss_bytes()
    payload = {
        "label": label,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", ""),
        "rss_max": _format_bytes(rss) if rss is not None else "unavailable",
        "gpu_memory": _gpu_memory_summary(),
    }
    if extra:
        payload.update(extra)
    print("[runtime] " + pprint.pformat(payload), flush=True)




def l1_loss(predicted_logits, true_tokens, valid=None):
    # Get the predicted tokens by taking the argmax over logits
    predicted_tokens = jnp.argmax(predicted_logits, axis=-1)
    
    # Calculate the L1 loss as the sum of absolute differences between predicted and true tokens
    loss = jnp.abs(predicted_tokens - true_tokens)
    
    # Mask the loss with the valid mask if provided
    if valid is not None:
        loss = loss * valid
    loss = jnp.mean(jnp.sum(loss, axis=-1))
    
    return loss

FLAGS, FLAGS_DEF = define_flags_with_default(
    modality='text',
    use_data_sharded_loader=True,
    seed=42,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=10,
    eval_log_freq = 10,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=VideoLLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    unseen_eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=VideoLLaMAConfig.get_default_config(),
    logger=tux.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    autoresume=False,
    delta_tokens=0,
    freeze=0,
    freeze_vision_params=False,
    mse_loss=1,
    runtime_log_steps=3,
    abort_on_nonfinite=True,
    diagnose_numerics=False,
    keep_last_milestones=0,
) 



def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = tux.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = tux.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    logger = tux.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)
    _runtime_log(
        "startup",
        {
            "jax_process_index": jax.process_index(),
            "jax_process_count": jax.process_count(),
            "jax_device_count": jax.device_count(),
            "jax_local_device_count": jax.local_device_count(),
            "jax_devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "mesh_dim": FLAGS.mesh_dim,
            "modality": FLAGS.modality,
            "total_steps": FLAGS.total_steps,
            "batch_size": getattr(FLAGS.train_dataset.json_delta_action_dataset, "batch_size", None),
        },
    )

    if jax.process_index() == 0:
        output_dir = logger.output_dir
    else:
        output_dir = os.path.join(logger.output_dir, logger.experiment_id)

    if FLAGS.modality == 'text':
        config_cls = LLaMAConfig
        llama_cls = FlaxLLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,text':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxVideoLLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,text,delta':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDeltaLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,action':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxActionLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,action,delta':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDeltaActionLaMAForCausalLMModule
    else:
        raise ValueError(f"Unsupported modality: {FLAGS.modality}")

    mesh_start = time.time()
    mesh = config_cls.get_jax_mesh(FLAGS.mesh_dim)
    node_info = config_cls.get_ranks_and_size(mesh)
    _runtime_log(
        "mesh_ready",
        {
            "elapsed_sec": round(time.time() - mesh_start, 3),
            "node_info": str(node_info),
        },
    )

    tokenizer = config_cls.get_tokenizer(FLAGS.tokenizer)
    dataset_start = time.time()
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer, node_info=node_info)
    _runtime_log(
        "dataset_ready",
        {
            "elapsed_sec": round(time.time() - dataset_start, 3),
            "dataset_type": FLAGS.train_dataset.type,
        },
    )
    if FLAGS.autoresume and tux.check_exists(output_dir):
        logging.info('Found existing output. Resuming dataset from latest checkpoint...')
        resume_path = f"{output_dir}/dataset.pkl"
        dataset.load_state_dict(tux.load_pickle(resume_path))
    elif FLAGS.load_dataset_state != '':
        dataset.load_state_dict(tux.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer, node_info=node_info)
        eval_iterator = iter(eval_dataset)
        unseen_eval_dataset = DatasetFactory.load_dataset(
            FLAGS.unseen_eval_dataset, dataset.tokenizer, node_info=node_info)
        unseen_eval_iterator = iter(unseen_eval_dataset)

    seq_length = dataset.seq_length
    _runtime_log(
        "dataset_config",
        {
            "seq_length": seq_length,
            "vocab_size": getattr(dataset, "vocab_size", None),
            "depth_feature_dir": getattr(FLAGS.train_dataset.json_delta_action_dataset, "depth_feature_data_dir", ""),
            "depth_feature_dim": getattr(FLAGS.train_dataset.json_delta_action_dataset, "depth_feature_dim", None),
        },
    )

    if FLAGS.load_llama_config != '':
        llama_config = config_cls.load_config(FLAGS.load_llama_config)
        updates = config_cls(**FLAGS.llama)
        llama_config.update(dict(
            remat_block=updates.remat_block,
            remat_attention=updates.remat_attention,
            remat_mlp=updates.remat_mlp,
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        llama_config = config_cls(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    model = llama_cls(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )
    use_depth_features = bool(
        getattr(FLAGS.train_dataset.json_delta_action_dataset, "depth_feature_data_dir", "")
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_mask(config_cls.get_weight_decay_exclusions()),
        None,
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        batch = 512
        if FLAGS.modality == 'text':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,text':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,text,delta':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                delta_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,action':
            depth_features = (
                jnp.zeros((batch, FLAGS.train_dataset.json_delta_action_dataset.depth_feature_dim), dtype=jnp.float32)
                if use_depth_features
                else None
            )
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                depth_features=depth_features,
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,action,delta':
            depth_features = (
                jnp.zeros((batch, FLAGS.train_dataset.json_delta_action_dataset.depth_feature_dim), dtype=jnp.float32)
                if use_depth_features
                else None
            )
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                delta_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                depth_features=depth_features,
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        else:
            raise ValueError(f"Unsupported modality: {FLAGS.modality}")
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def freeze_named_grads(grads, frozen_names):
        flat_grads = flatten_dict(flax.core.frozen_dict.unfreeze(grads))
        for path, value in flat_grads.items():
            if any(name in path for name in frozen_names):
                flat_grads[path] = jax.tree_util.tree_map(jnp.zeros_like, value)
        return flax.core.frozen_dict.freeze(unflatten_dict(flat_grads))

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        def loss_and_accuracy(params):
            if FLAGS.modality == 'text':
                logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                loss, acc = cross_entropy_loss_and_accuracy(
                    logits, 
                    batch['target_tokens'],
                    batch['loss_masks']
                )
                metrics = dict(acc=acc)
                return loss, metrics
            elif FLAGS.modality == 'vision,text':
                vision_logits, text_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where(batch['target_vision_masks'], 0, batch['target_tokens']),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'])
                )
                loss = text_loss
                
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                )
            elif FLAGS.modality == 'vision,text,delta':
                vision_logits, text_logits, delta_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    batch['input_delta_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                    delta_logits, 
                    jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_delta_masks']
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']), batch['target_tokens'], 0),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_delta_masks']))
                )
                loss = 0.99 * delta_loss + 0.01 * text_loss 
                
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    delta_loss=delta_loss,
                    delta_acc=delta_acc,
                )
            elif FLAGS.modality == 'vision,action':
                vision_logits, text_logits, action_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    batch['input_action_masks'],
                    depth_features=batch.get('depth_features'),
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                action_loss, action_acc = cross_entropy_loss_and_accuracy(
                    action_logits, 
                    jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_action_masks']
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_action_masks']), batch['target_tokens'], 0),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_action_masks']))
                )
                loss = action_loss
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    action_loss=action_loss,
                    action_acc=action_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                )
            elif FLAGS.modality == 'vision,action,delta':
                vision_logits, text_logits, delta_logits, action_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    batch['input_delta_masks'],
                    batch['input_action_masks'],
                    depth_features=batch.get('depth_features'),
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                    delta_logits, 
                    jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_delta_masks']
                )
                action_loss, action_acc = cross_entropy_loss_and_accuracy(
                    action_logits, 
                    jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_action_masks']
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']) * (1.0 - batch['target_action_masks']), batch['target_tokens'], 0),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks']) * (1.0 - batch['target_delta_masks']) * (1.0 - batch['target_action_masks']),
                )
                loss = action_loss
                
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    delta_loss=delta_loss,
                    delta_acc=delta_acc,
                    action_loss=action_loss,
                    action_acc=action_acc,
                )
                if FLAGS.diagnose_numerics:
                    # Targeted activation checks only. Unlike the old debug path,
                    # these do not scan the 7B parameter/gradient trees.
                    for name, value in (
                        ("depth_features", batch.get("depth_features")),
                        ("action_logits", action_logits),
                        ("text_logits", text_logits),
                        ("delta_logits", delta_logits),
                    ):
                        if value is None:
                            continue
                        metrics[f"{name}_nonfinite"] = jnp.sum(
                            ~jnp.isfinite(value)
                        )
                        finite_value = jnp.where(
                            jnp.isfinite(value),
                            value.astype(jnp.float32),
                            0.0,
                        )
                        metrics[f"{name}_finite_absmax"] = jnp.max(
                            jnp.abs(finite_value)
                        )
                    metrics.update(
                        action_loss_mask_sum=jnp.sum(
                            batch["loss_masks"] * batch["target_action_masks"]
                        ),
                        text_loss_mask_sum=jnp.sum(
                            batch["loss_masks"]
                            * (1.0 - batch["target_vision_masks"])
                            * (1.0 - batch["target_delta_masks"])
                            * (1.0 - batch["target_action_masks"])
                        ),
                    )
            else:
                raise ValueError(f"Unsupported modality: {FLAGS.modality}")
            return loss, metrics 
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, loss_metrics), grads = grad_fn(train_state.params)
        if FLAGS.freeze_vision_params:
            grads = freeze_named_grads(grads, ("vte", "vision_head"))


        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            param_norm=global_norm(train_state.params),
            gradient_norm=global_norm(grads),
            **loss_metrics
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        if FLAGS.modality == 'text':
            logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            loss, acc = cross_entropy_loss_and_accuracy(
                logits, 
                batch['target_tokens'],
                batch['loss_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_acc=acc,
            )
        elif FLAGS.modality == 'vision,text':
            vision_logits, text_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where(batch['target_vision_masks'], 0, batch['target_tokens']),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'])
            )
            loss = text_loss
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
            )
        elif FLAGS.modality == 'vision,text,delta':
            vision_logits, text_logits, delta_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_delta_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                delta_logits, 
                jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_delta_masks']
            )
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']), batch['target_tokens'], 0),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_delta_masks']))
            )
            loss = 0.99 * delta_loss + 0.01 * text_loss 
            # loss = delta_loss
            # TODO: add pheanki reconstruction result for validation
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
                eval_delta_accuracy=delta_acc,
                eval_delta_loss=delta_loss,
            )
        elif FLAGS.modality == 'vision,action':
            vision_logits, text_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_action_masks'],
                depth_features=batch.get('depth_features'),
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_action_masks']), batch['target_tokens'], 0),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_action_masks']))
            )
            loss = action_loss
            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_action_accuracy=action_acc,
                eval_action_loss=action_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
                eval_action_l1_loss=action_l1_loss,
            )
        elif FLAGS.modality == 'vision,action,delta':
            vision_logits, text_logits, delta_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_delta_masks'],
                batch['input_action_masks'],
                depth_features=batch.get('depth_features'),
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                delta_logits, 
                jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_delta_masks']
            )
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']) * (1.0 - batch["target_action_masks"]), batch['target_tokens'], 0),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks']) * (1.0 - batch['target_delta_masks']) * (1.0 - batch['target_action_masks']),
            )
            loss = action_loss

            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )


            # TODO: add pheanki reconstruction result for validation
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
                eval_delta_accuracy=delta_acc,
                eval_delta_loss=delta_loss,
                eval_action_accuracy=action_acc,
                eval_action_loss=action_loss,
                eval_action_l1_loss=action_l1_loss,
            )
        return rng_generator(), metrics
    
    def unseen_eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        if FLAGS.modality == 'text':
            logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            loss, acc = cross_entropy_loss_and_accuracy(
                logits, 
                batch['target_tokens'],
                batch['loss_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_acc=acc,
            )
        elif FLAGS.modality == 'vision,action':
            vision_logits, text_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_action_masks'],
                depth_features=batch.get('depth_features'),
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            loss = action_loss

            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )

            metrics = dict(
                unseen_eval_loss=loss,
                unseen_eval_action_accuracy=action_acc,
                unseen_eval_action_loss=action_loss,
                unseen_eval_action_l1_loss=action_l1_loss,
            )
        elif FLAGS.modality == 'vision,action,delta':
            vision_logits, text_logits, delta_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_delta_masks'],
                batch['input_action_masks'],
                depth_features=batch.get('depth_features'),
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            loss = action_loss

            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )

            metrics = dict(
                unseen_eval_loss=loss,
                unseen_eval_action_accuracy=action_acc,
                unseen_eval_action_loss=action_loss,
                unseen_eval_action_l1_loss=action_l1_loss,
            )
        return rng_generator(), metrics

    _runtime_log("eval_shape_start")
    eval_shape_start = time.time()
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    _runtime_log("eval_shape_done", {"elapsed_sec": round(time.time() - eval_shape_start, 3)})
    partition_start = time.time()
    train_state_partition = match_partition_rules(
        config_cls.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    _runtime_log("partition_ready", {"elapsed_sec": round(time.time() - partition_start, 3)})
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )
    params_checkpointer_config = copy.deepcopy(FLAGS.checkpointer)
    params_checkpointer_config.save_optimizer_state = False
    params_checkpointer = StreamingCheckpointer(
        params_checkpointer_config, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if FLAGS.use_data_sharded_loader:
        batch_spec = PS(('dp', 'fsdp'), 'sp')
    else:
        batch_spec = PS()
    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    sharded_unseen_eval_step = pjit(
        unseen_eval_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None, max_buffer_size=0):
        if shard_fns is not None:
            shard_fns = flatten_dict(
                to_state_dict(shard_fns)
            )
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flattend_train_state = {}
        with open_file(path) as fin:
            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=max_buffer_size)
            for key, value in unpacker:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix):]
                    else:
                        continue
                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    tensor = shard_fns[key](tensor)
                flattend_train_state[key] = tensor

        if target is not None:
            flattened_target = flatten_dict(
                to_state_dict(target), keep_empty_nodes=True
            )
            missing_keys = []
            for key, value in flattened_target.items():
                key_name = "/".join(str(part) for part in key)
                if (
                    key in flattend_train_state
                    and hasattr(flattend_train_state[key], "shape")
                    and hasattr(value, "shape")
                ):
                    loaded_shape = tuple(flattend_train_state[key].shape)
                    target_shape = tuple(value.shape)
                    if loaded_shape != target_shape:
                        if key_name == "action_head/kernel":
                            # Switching project (4096-D) <-> concat (5120-D)
                            # intentionally changes only the action-head input
                            # dimension. Reinitialize this head from the Stage-2
                            # checkpoint while preserving the backbone.
                            del flattend_train_state[key]
                            _runtime_log(
                                "checkpoint_reinitialize_shape_mismatch",
                                {
                                    "key": key_name,
                                    "loaded_shape": loaded_shape,
                                    "target_shape": target_shape,
                                },
                            )
                        else:
                            raise ValueError(
                                f"Checkpoint shape mismatch for {key_name}: "
                                f"loaded={loaded_shape}, expected={target_shape}"
                            )
                if key not in flattend_train_state and value == empty_node:
                    flattend_train_state[key] = value
                elif key not in flattend_train_state:
                    missing_index = len(missing_keys)
                    missing_keys.append(key_name)
                    rng = jax.random.fold_in(
                        jax.random.PRNGKey(FLAGS.seed),
                        missing_index,
                    )

                    # These Stage-3 tensors are intentionally absent from the
                    # Stage-2 checkpoint. Initialize them exactly as declared by
                    # Flax Embed/Dense in delta_llama_action.py, rather than
                    # applying fan-in-dependent LeCun initialization in the
                    # checkpoint loader. Distinct keys avoid correlated tensors.
                    if key_name in (
                        "transformer/ate/embedding",
                        "action_head/kernel",
                        "depth_action_proj/kernel",
                    ):
                        initializer = jax.nn.initializers.normal(
                            stddev=llama_config.initializer_range
                        )
                    elif key_name.endswith("/bias"):
                        initializer = jax.nn.initializers.zeros
                    else:
                        initializer = jax.nn.initializers.lecun_normal()

                    tensor = initializer(
                        rng,
                        value.shape,
                        dtype=value.dtype,
                    )
                    flattend_train_state[key] = tensor
            if missing_keys:
                _runtime_log(
                    "checkpoint_initialized_missing_params",
                    {
                        "count": len(missing_keys),
                        "keys": missing_keys,
                        "stage3_initializer": (
                            "normal(stddev="
                            f"{llama_config.initializer_range}, distinct_rngs)"
                        ),
                    },
                )


        train_state = unflatten_dict(flattend_train_state)
        if target is None:
            return train_state

        return from_state_dict(target, train_state)

    def _cleanup_old_milestones(current_step):
        keep = int(FLAGS.keep_last_milestones)
        if keep <= 0 or jax.process_index() != 0:
            return

        candidates = []
        prefixes = ("streaming_params_", "streaming_train_state_", "metadata_", "dataset_")
        for name in os.listdir(output_dir):
            for prefix in prefixes:
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix):]
                if suffix.isdigit():
                    candidates.append((int(suffix), os.path.join(output_dir, name)))
                break

        protected_steps = sorted({step for step, _ in candidates if step <= current_step}, reverse=True)[:keep]
        protected_steps = set(protected_steps)
        removed = []
        for step, path in candidates:
            if step in protected_steps:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
            removed.append(os.path.basename(path))

        if removed:
            _runtime_log(
                "checkpoint_cleanup_old_milestones",
                {
                    "current_step": current_step,
                    "keep_last_milestones": keep,
                    "removed": removed,
                },
            )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )
        if getattr(FLAGS.checkpointer, "save_optimizer_state", False):
            params_checkpointer.save_all(
                train_state=train_state,
                gather_fns=gather_fns,
                metadata=metadata,
                dataset=dataset.get_state_dict(),
                milestone=milestone,
            )
        if milestone:
            _cleanup_old_milestones(step)

    with mesh:
        train_state, restored_params = None, None

        if FLAGS.autoresume and tux.check_exists(output_dir):
            logging.info('Found existing output. Resuming model from latest checkpoint...')
            resume_path = f"trainstate::{output_dir}/streaming_train_state"
            _runtime_log("checkpoint_load_start", {"checkpoint": resume_path})
            load_start = time.time()
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                resume_path, train_state_shapes, shard_fns, max_buffer_size=32 * 2 ** 30
            )
            _runtime_log("checkpoint_load_done", {"elapsed_sec": round(time.time() - load_start, 3)})
        elif FLAGS.load_checkpoint != '':
            params_target = train_state_shapes.params['params']
            params_shard_fns = shard_fns.params['params']
            load_type, load_path = FLAGS.load_checkpoint.split('::', 1)
            train_state = None
            restored_params = None
            _runtime_log("checkpoint_load_start", {"load_type": load_type, "checkpoint": load_path})
            load_start = time.time()
            if load_type == 'trainstate':
                train_state = load_checkpoint(
                    path=load_path,
                    target=train_state_shapes,
                    shard_fns=shard_fns,
                    max_buffer_size=32 * 2 ** 30,
                )
            elif load_type == 'trainstate_params':
                # Load the params part of the train state in the streaming format
                restored_params = load_checkpoint(
                    path=load_path,
                    target=params_target,
                    shard_fns=params_shard_fns,
                    remove_dict_prefix=('params', 'params'),
                    max_buffer_size=32 * 2 ** 30,
                )
                restored_params = flax.core.frozen_dict.freeze(
                    {'params': restored_params}
                )
            elif load_type == 'params':
                # Load the params in the streaming format
                restored_params = load_checkpoint(
                    path=load_path,
                    target=params_target,
                    shard_fns=params_shard_fns,
                    max_buffer_size=32 * 2 ** 30
                )
                restored_params = flax.core.frozen_dict.freeze(
                    {'params': restored_params}
                )
            _runtime_log("checkpoint_load_done", {"elapsed_sec": round(time.time() - load_start, 3)})

        if train_state is None and restored_params is None:
            # Initialize from scratch
            _runtime_log("train_state_init_start", {"source": "scratch"})
            init_start = time.time()
            train_state = sharded_init_fn(next_rng())
            _runtime_log("train_state_init_done", {"elapsed_sec": round(time.time() - init_start, 3)})
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            _runtime_log("train_state_init_start", {"source": "restored_params"})
            init_start = time.time()
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params
            _runtime_log("train_state_init_done", {"elapsed_sec": round(time.time() - init_start, 3)})

        start_step = int(jax.device_get(train_state.step))
        _runtime_log("train_state_ready", {"start_step": start_step})

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)
        dataset_iterator = iter(dataset)
        runtime_log_steps = max(int(FLAGS.runtime_log_steps), 0)
        _runtime_log(
            "train_loop_start",
            {
                "start_step": start_step,
                "total_steps": FLAGS.total_steps,
                "runtime_log_steps": runtime_log_steps,
            },
        )
        for step in step_counter:
            fetch_start = time.time()
            try:
                batch, dataset_metrics = next(dataset_iterator)
            except StopIteration:
                break
            fetch_elapsed = time.time() - fetch_start
            if step < start_step + runtime_log_steps:
                _runtime_log(
                    "batch_ready",
                    {
                        "step": step,
                        "fetch_sec": round(fetch_elapsed, 3),
                        "batch": _batch_summary(batch),
                        "dataset_metrics": pprint.pformat(dataset_metrics),
                    },
                )
            train_step_start = time.time()
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch 
            )
            if step < start_step + runtime_log_steps:
                metrics_host = jax.device_get(metrics)
                _runtime_log(
                    "train_step_done",
                    {
                        "step": step,
                        "step_sec": round(time.time() - train_step_start, 3),
                        "metrics": pprint.pformat(metrics_host),
                    },
                )
            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0 and step % FLAGS.eval_log_freq == 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metrics = jax.device_get(eval_metrics)
                        eval_batch, _ = next(unseen_eval_iterator)
                        sharded_rng, eval_metrics2 = sharded_unseen_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metrics2 = jax.device_get(eval_metrics2)
                        # concat two dict 
                        eval_metrics.update(eval_metrics2)
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                if FLAGS.abort_on_nonfinite:
                    nonfinite_metrics = {
                        key: value
                        for key, value in log_metrics.items()
                        if key in (
                            "loss",
                            "action_loss",
                            "text_loss",
                            "gradient_norm",
                            "param_norm",
                        )
                        and not np.all(np.isfinite(np.asarray(value)))
                    }
                    if nonfinite_metrics:
                        raise FloatingPointError(
                            "Stopping training because scalar metrics became "
                            f"non-finite at step {step}: "
                            f"{pprint.pformat(nonfinite_metrics)}"
                        )
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    run(main)
