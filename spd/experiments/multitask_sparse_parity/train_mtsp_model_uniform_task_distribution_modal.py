# n_control_bits = 10
# n_task_bits = 30
# n_xored_bits = 4
# batch_sz = 64
import hashlib
import json
from dataclasses import asdict, dataclass

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "coolname", "safetensors", "packaging")
    .add_local_python_source("multitask_sparse_parity")
    .add_local_python_source("storer")
    .add_local_python_source("modal_utils")
)
app = modal.App(
    "example-custom-container",
    image=image,
    volumes={"/mtsp_results": modal.Volume.from_name("mtsp_results", create_if_missing=True)},
)

# import matplotlib.pyplot as plt
import numpy as np

# import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    d_mlp: int = 256
    seed: int = 0
    steps: int = 100_000
    lr: float = 1e-3
    n_control_bits: int = 20
    n_task_bits: int = 30
    n_xored_bits: int = 4
    task_distribution_decay_rate: float = 0
    batch_sz: int = 64
    code_version: str = "v1"

    def cache_key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


# for d_mlp in [32, 64, 128, 256, 512]:


@app.function()
def train_model(config):
    storer = Storer("/modal_volume/mtsp_results/metrics/")
    storer.add_datestamp_prefix()
    storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{config.cache_key()}")
    print(storer)

    torch.manual_seed(config.seed)
    model = MultitaskSparseParityModel(
        d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
    )
    # batch_sz = 64
    # steps = 30000
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses_by_step_and_task = []

        train_dataloader = DataLoader(dataset, batch_size=None)

        for step, (task_ids, task_bits, parity) in enumerate(train_dataloader):
            optimizer.zero_grad()

            logits = model((task_ids, task_bits, ()))
            loss = F.cross_entropy(logits, parity)

            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == config.steps - 1:
                print(f"{step=}", loss.item())

                losses_by_task_for_step = []
                with torch.no_grad():
                    task_ids, task_bits, parity = dataset.get_batch(batch_sz=64)
                    logits = model((task_ids, task_bits, ()))
                    val_loss = F.cross_entropy(logits, parity)
                    losses_by_task_for_step.append(val_loss.item())

                    for i in range(config.n_control_bits):
                        task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
                            batch_sz=64, task_ids=torch.tensor(i)
                        )
                        logits = model((task_ids, task_bits, ()))
                        loss = F.cross_entropy(logits, parity)
                        # print(i, loss.item())
                        losses_by_task_for_step.append(loss.item())

                losses_by_step_and_task.append(losses_by_task_for_step)

        losses_by_step_and_task = np.array(losses_by_step_and_task)

        state = model.state_dict()
        storer.write(key, state)
        storer.write("losses_by_step_and_task", losses_by_step_and_task)

    model.load_state_dict(storer.read(key))
    losses_by_step_and_task = storer.read("losses_by_step_and_task")
    torch.manual_seed(0)
    task_ids, task_bits, parity = dataset.get_batch(batch_sz=1024)
    logits = model((task_ids, task_bits, ()))
    val_loss = F.cross_entropy(logits, parity)
    print(val_loss)


# from pathlib import Path

# root_path = "/modal_volume/mtsp_results/modal_volume/metrics/"
# root_path = Path(root_path)
# if root_path.parts[:2] == ("/", "modal_volume"):
#     local = False
#     on_modal = True
#     if local:
#         root_path = Path("./modal_volume_cache", *root_path.parts[2:])
#     if on_modal:
#         root_path = Path("/", *root_path.parts[2:])


# print(root_path)

# d_mlp = 64
# for d_mlp in [32, 64, 128, 256, 512]:
#     storer = Storer("/modal_volume/mtsp_results/metrics/")
#     config = TrainConfig(d_mlp=d_mlp)

#     storer.add_datestamp_prefix()
#     storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{config.cache_key()}")
#     # print(storer)
#     torch.manual_seed(config.seed)
#     model = MultitaskSparseParityModel(
#         d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
#     )
#     dataset = MultitaskSparseParityDataset(
#         n_control_bits=config.n_control_bits,
#         n_task_bits=config.n_task_bits,
#         n_xored_bits=config.n_xored_bits,
#         task_distribution_decay_rate=config.task_distribution_decay_rate,
#         batch_sz=config.batch_sz,
#         size=config.steps,
#     )
#     storer.add_prefix(f"model/{config.steps}")
#     key = "model"
#     assert storer.exists(key), (d_mlp, storer)
#     model.load_state_dict(storer.read(key))
#     losses_by_step_and_task = storer.read("losses_by_step_and_task")
#     torch.manual_seed(0)
#     task_ids, task_bits, parity = dataset.get_batch(batch_sz=1024)
#     logits = model((task_ids, task_bits, ()))
#     val_loss = F.cross_entropy(logits, parity)
#     print(val_loss)

#     losses_by_task = []
#     for i in range(config.n_control_bits):
#         task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
#             batch_sz=1000, task_ids=torch.tensor(i)
#         )
#         logits = model((task_ids, task_bits, ()))
#         loss = F.cross_entropy(logits, parity)
#         losses_by_task.append(loss.item())
#     losses_by_task = torch.tensor(losses_by_task)
#     print((losses_by_task < torch.log(torch.tensor(1.5))).float().mean())


# config = TrainConfig(d_mlp=64)
# train_model(config)
import itertools


@app.local_entrypoint()
def main():
    # config = TrainConfig(d_mlp=64)
    # train_model.remote(config)

    d_mlps = [32, 64, 128, 256, 512]
    seeds = list(range(5))

    configs = [TrainConfig(d_mlp=d_mlp, seed=seed) for d_mlp in d_mlps for seed in seeds]
    # print(configs)
    # configs = [TrainConfig(d_mlp=d_mlp) for d_mlp in ]
    for result in train_model.map(configs):
        pass


# x = np.arange(losses_by_step_and_task.shape[0]) * 100

# n_lines = losses_by_step_and_task.shape[1] - 1
# plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
# plt.plot(
#     x,
#     losses_by_step_and_task[:, 1:],
#     label=list(np.arange(config.n_control_bits)),
#     linewidth=0.5,
# )

# plt.plot(x, losses_by_step_and_task[:, 0], label="overall", color="red", linewidth=1)
# plt.legend()
# # plt.xscale("log")
# plt.show()


# wandb.init(project="multitask-sparse-parity")

# for step_idx, step in enumerate(range(0, steps, 100)):
#     log_dict = {"step": step, "loss/overall": losses_by_step_and_task[step_idx, 0]}
#     for task_id in range(n_control_bits):
#         log_dict[f"loss/task_{task_id}"] = losses_by_step_and_task[step_idx, task_id + 1]
#     wandb.log(log_dict)

# torch.save(model.state_dict(), "model.pt")
# artifact = wandb.Artifact("multitask-sparse-parity-model", type="model")
# artifact.add_file("model.pt")
# wandb.log_artifact(artifact)

# wandb.finish()
