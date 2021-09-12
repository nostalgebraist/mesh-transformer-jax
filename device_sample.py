import argparse
import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--base-model-path", type=str, default=None, help="Base model path if using adapters")

    parser.add_argument("--temp", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]
    use_adapters = params.get("use_adapters", False)

    if use_adapters and (args.base_model_path is None):
        raise ValueError(f"If using adapters, you must pass --base-model-paths")

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    gptj_model_dir = model_dir
    adapter_model_dir = None

    if use_adapters:
        gptj_model_dir = args.base_model_path
        adapter_model_dir = model_dir

    try:
        with open(f"gs://{bucket}/{gptj_model_dir}/meta.json", "r") as f:
            meta = json.load(f)
        ckpt_step = meta["checkpoints"][-1]
        print(f"using checkpoint {ckpt_step}")
        gptj_ckpt_path = f"gs://{bucket}/{gptj_model_dir}/step_{ckpt_step}/"
    except:
        # this try/except exists for convenience given the way i have my gcs bucket set up -nost
        print(f"couldn't load meta.json, interpreting {gptj_model_dir} as a direct ckpt path")
        gptj_ckpt_path = gptj_model_dir

    if use_adapters:
        with open(f"gs://{bucket}/{adapter_model_dir}/meta.json", "r") as f:
            meta = json.load(f)

        ckpt_step = meta["checkpoints"][-1]
        print(f"using adapter checkpoint {ckpt_step}")

        adapter_ckpt_path = f"gs://{bucket}/{adapter_model_dir}/step_{ckpt_step}/"

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, gptj_ckpt_path, devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        if use_adapters:
            start = time.time()
            network.state = read_ckpt(network.state, adapter_ckpt_path, devices.shape[1], adapter_ckpt=True)
            print(f"adapters loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        while True:
            context = input("Type input:")
            tokens = tokenizer.encode(context)

            start = time.time()

            provided_ctx = len(tokens)
            pad_amount = seq - provided_ctx

            padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
            batched_tokens = np.array([padded_tokens] * total_batch)
            length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

            output = network.generate(batched_tokens, length, 512, {"top_p": np.ones(total_batch) * args.top_p,
                                                                    "temp": np.ones(total_batch) * args.temp})

            for idx, o in enumerate(output[1][0][:, :, 0]):
                print(f"sample {idx}: {repr(tokenizer.decode(o))}")

            print(f"completion done in {time.time() - start:06}s")
