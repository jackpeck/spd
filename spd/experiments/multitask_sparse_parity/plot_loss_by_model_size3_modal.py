import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import modal
import numpy as np
import pandas as pd
import torch

# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer

# import wandb
from torch.utils.data import DataLoader
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "coolname", "safetensors", "packaging", "matplotlib", "pandas")
    .add_local_python_source("multitask_sparse_parity")
    .add_local_python_source("storer")
    .add_local_python_source("modal_utils")
    .add_local_python_source("train_mtsp_model_uniform_task_distribution_modal")
)
app = modal.App(
    "example-custom-container",
    image=image,
    volumes={"/mtsp_results": modal.Volume.from_name("mtsp_results", create_if_missing=True)},
)

rows = []


@app.function(timeout=360)
def f(config):
    storer = Storer("/modal_volume/mtsp_results/metrics/")

    storer.add_datestamp_prefix()
    storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{config.cache_key()}")
    torch.manual_seed(config.seed)
    model = MultitaskSparseParityModel(
        d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
    )
    dataset = MultitaskSparseParityDataset(
        n_control_bits=config.n_control_bits,
        n_task_bits=config.n_task_bits,
        n_xored_bits=config.n_xored_bits,
        task_distribution_decay_rate=config.task_distribution_decay_rate,
        batch_sz=config.batch_sz,
        size=config.steps,
    )
    storer.add_prefix(f"model/{config.steps}")
    key = "model"
    if not storer.exists(key):
        print("key not found, skipping", (config.d_mlp, storer))
        return

    model.load_state_dict(storer.read(key))
    torch.manual_seed(0)
    task_ids, task_bits, parity = dataset.get_batch(batch_sz=1024)
    logits = model((task_ids, task_bits, ()))
    val_loss = F.cross_entropy(logits, parity)
    print(val_loss)

    key = "task_learnt_proportion"

    if not storer.exists(key):
        losses_by_task = []
        for i in range(config.n_control_bits):
            task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
                batch_sz=1024, task_ids=torch.tensor(i)
            )
            logits = model((task_ids, task_bits, ()))
            loss = F.cross_entropy(logits, parity)
            losses_by_task.append(loss.item())
        losses_by_task = torch.tensor(losses_by_task)
        task_learnt_proportion = (losses_by_task < torch.log(torch.tensor(2)) / 2).float().mean()
        storer.write(key, task_learnt_proportion)

    # task_learnt_proportion = storer.read(key)

    # row = {
    #     **dataclasses.asdict(config),
    #     "val_loss": val_loss.item(),
    #     "task_learnt_proportion": task_learnt_proportion,
    # }

    # rows.append(row)


d_mlps = list(np.geomspace(16, 512, 11).round().astype(int))
configs = []
for seed in range(5):
    for d_mlp in d_mlps:
        config = TrainConfig(d_mlp=d_mlp, seed=seed)
        configs.append(config)


@app.local_entrypoint()
def main():
    for result in f.map(configs):
        pass


def plot():
    # for seed in range(5):
    #     for d_mlp in d_mlps:

    for config in configs:
        storer = Storer("/modal_volume/mtsp_results/metrics/")

        storer.add_datestamp_prefix()
        storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{config.cache_key()}")
        torch.manual_seed(config.seed)
        model = MultitaskSparseParityModel(
            d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
        )
        dataset = MultitaskSparseParityDataset(
            n_control_bits=config.n_control_bits,
            n_task_bits=config.n_task_bits,
            n_xored_bits=config.n_xored_bits,
            task_distribution_decay_rate=config.task_distribution_decay_rate,
            batch_sz=config.batch_sz,
            size=config.steps,
        )
        storer.add_prefix(f"model/{config.steps}")
        key = "model"
        if not storer.exists(key):
            print("key not found, skipping", (config.d_mlp, storer))
            continue

        model.load_state_dict(storer.read(key))
        torch.manual_seed(0)
        task_ids, task_bits, parity = dataset.get_batch(batch_sz=1024)
        logits = model((task_ids, task_bits, ()))
        val_loss = F.cross_entropy(logits, parity)
        print(val_loss)

        key = "task_learnt_proportion"

        assert storer.exists(key)
        task_learnt_proportion = storer.read(key)
        row = {
            **dataclasses.asdict(config),
            "val_loss": val_loss.item(),
            "task_learnt_proportion": task_learnt_proportion,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    grouped = df.groupby("d_mlp")["task_learnt_proportion"]
    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()

    plt.figure(dpi=150)
    plt.errorbar(
        means.index,
        means.values,
        yerr=stds.values * 2 / np.sqrt(counts.values),
        fmt="o-",
        capsize=4,
    )
    plt.xlabel("d_mlp")
    plt.ylabel("proportion of tasks under 0.5 bits prediction error")
    plt.xlim(0, 250)
    # plt.yscale("log")
    default_config = TrainConfig()
    plt.title(
        f"n_control_bits={default_config.n_control_bits}, n_task_bits={default_config.n_task_bits}"
    )

    hash = hashlib.sha256(
        json.dumps(
            {
                "default_config_hash": TrainConfig().cache_key(),
                "seeds": 5,
                "d_mlps": d_mlps,
            },
            default=int,
        ).encode()
    ).hexdigest()[:8]
    plt.savefig(f"plots/{hash}-proportion of tasks under 0.5 bits prediction error.png")
    plt.show()


if __name__ == "__main__":
    plot()
