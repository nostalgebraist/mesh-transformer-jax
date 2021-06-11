import argparse
import json
import time

import jax
import numpy as np
import optax

import wandb
from tqdm import tqdm


from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from tfrecord_loader import TFRecordNewInputs
from smart_open import open
from google.cloud import storage
from google.cloud.exceptions import NotFound

from mesh_transformer.util import clip_by_global_norm, additive_weight_decay


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--tune-config", type=str, default=None, help="Config file for base model to finetune")

    args = parser.parse_args()
    return args


def save(step, bucket, path, mp, aux=None, init=False, overwrite=False, keep_n=3, delete_old=True):
    assert path
    client = storage.Client()

    if aux is None:
        aux = {}

    if init:
        # check existing checkpoint folder does not exist, and delete it if it does
        for blob in client.list_blobs(bucket, prefix=f"{path}/"):
            assert overwrite
            assert path in blob.name
            blob.delete()

        # create metadata file
        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    for shard_id in range(mp):
        write_ckpt(f"gs://{bucket}/{path}/step_{step}/", shard_id)

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                # print(f"deleting {blob.name}")
                assert path in blob.name
                blob.delete()
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        json.dump(meta, f)


def train_step(network, data):
    inputs = {
        "obs": data[:, :, :-1],
        "target": data[:, :, 1:],
    }

    loss, last_loss = network.train(inputs)

    return np.array(loss).mean(), np.array(last_loss).mean()


def eval_step(network, data):
    inputs = {
        "obs": data[:, :, :-1],
        "target": data[:, :, 1:],
    }

    out = network.eval(inputs)
    loss = out["loss"]

    return np.array(loss).mean()


if __name__ == "__main__":
    args = parse_args()
    tuning = args.tune_config is not None
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    eval_tasks = params["eval_harness_tasks"]
    total_steps = params["total_steps"]

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]
    step_shift = params.get("step_shift", 0)

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr, step_shift))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    # pick initial ckpt - based on tuning vs train from scratch

    if tuning:
        base_params = json.load(open(args.tune_config))
        initial_ckpt_bucket = base_params["bucket"]
        initial_ckpt_model_dir = base_params["model_dir"]
    else:
        initial_ckpt_bucket = bucket
        initial_ckpt_model_dir = model_dir

    initial_ckpt_path = f"gs://{initial_ckpt_bucket}/{initial_ckpt_model_dir}"
    meta_path = f"{initial_ckpt_path}/meta.json"

    ckpt_step = None
    initial_ckpt_state_path = None
    meta = None
    train_loader = None
    step = 0

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            ckpt_step = meta["checkpoints"][-1]
            initial_ckpt_state_path = f"{initial_ckpt_path}/step_{ckpt_step}/"
            print(f"state will be restored from checkpoint {ckpt_step}")

            if not tuning:
                step = ckpt_step
    except NotFound:
        if tuning:
            # load required for tuning
            msg = f"Couldn't find a checkpoint for config {args.tune_config}."
            msg += f" This file should exist, but doesn't: {meta_path}"
            raise ValueError(msg)

        # no checkpoint, start at zero
        print(f"No checkpoint to load at {initial_ckpt_path}. Training from scratch.")

    if meta and not tuning:
        aux = meta['aux']
        train_loader = aux.get("train_loader", None)

    # set up datasets

    tpu_size = jax.device_count()

    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                          gradient_accumulation_steps,
                                          per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'],
                                      restore_state=train_loader)

    global_val_batch = per_replica_batch * tpu_size // cores_per_replica

    val_sets = {}

    for k, v in params['val_set'].items():
        val_sets[k] = TFRecordNewInputs(f"data/{v}",
                                        batch_size=(global_val_batch,),
                                        sample_size=seq)

    # tok/sec metrics
    windows_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    tokens_per_step = params['seq'] * windows_per_step

    # load + run
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        if initial_ckpt_state_path:
            start = time.time()
            network.state = read_ckpt(network.state, initial_ckpt_state_path, devices.shape[1])
            print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        start = time.time()
        train_step(train_dataset.get_samples())
        step += 1
        print(f"Train fn compiled in {time.time() - start:.06}s")

        start = time.time()
        for val_set in val_sets.values():
            eval_step(val_set.get_samples())
            val_set.reset()
        print(f"Eval fn compiled in {time.time() - start:.06}s")

        wandb.init(project='mesh-transformer-jax', name=params["name"], config=params)

        while True:
            if (step % ckpt_every == 0 and step) or step == total_steps:
                save(step, bucket, model_dir,
                     mp=cores_per_replica,
                     aux={"train_loader": train_dataset.get_state()},
                     init=(step == 0),
                     delete_old=True,
                     )

                if step == total_steps:
                    print("training completed!")
                    exit()

            if step % val_every == 1:  # 1 because we've already taken a step to compile train fn
                for name, val_set in val_sets.items():
                    val_loss = []
                    for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                     desc=f"validation for step {step}, set {name}",
                                     total=val_batches):
                        val_loss.append(eval_step(i))
                    val_set.reset()

                    val_loss = np.array(val_loss).mean()
                    print(f"validation loss for step {step}, set {name}: {val_loss}")

                    wandb.log({f'val/loss_{name}': float(val_loss)}, step)

            start = time.time()
            loss, last_loss = train_step(train_dataset.get_samples())
            step += 1

            steps_per_sec = 1 / (time.time() - start)
            tokens_per_sec = tokens_per_step * steps_per_sec
