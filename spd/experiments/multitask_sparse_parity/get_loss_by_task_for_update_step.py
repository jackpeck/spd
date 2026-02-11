import hashlib
import json
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from torch.utils.data import DataLoader

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


@dataclass(frozen=True)
class TrainConfig:
    d_mlp: int = 64
    seed: int = 1
    steps: int = 80000
    lr: float = 1e-3
    n_control_bits: int = 10
    n_task_bits: int = 30
    n_xored_bits: int = 4
    task_distribution_decay_rate: float = 0
    batch_sz: int = 32
    code_version: str = "v1"
    eval_batch_sz: int = 1024
    eval_interval: int = 100

    def cache_key(self):
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                default=int,
            ).encode()
        ).hexdigest()[:16]


config = TrainConfig()


storer = Storer("/modal_volume/mtsp_results/metrics/")
storer.add_datestamp_prefix()
storer.add_prefix(f"get_loss_by_task_for_update_step/v11/{config.cache_key()}")
print(storer)

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
# key = "model"
# if not storer.exists(key):
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

loss_gradients_by_task = []
losses_by_task = []
losses_overall = []

train_dataloader = DataLoader(dataset, batch_size=None)

for step, (task_ids, task_bits, parity) in enumerate(train_dataloader):
    optimizer.zero_grad()

    logits = model((task_ids, task_bits, ()))
    loss = F.cross_entropy(logits, parity)

    old_params = {name: p.detach().clone() for name, p in model.named_parameters()}

    loss.backward()
    optimizer.step()

    param_step = {name: p - old_params[name] for name, p in model.named_parameters()}

    loss_gradients_by_task_for_step = []
    losses_by_task_for_step = []

    if step % config.eval_interval == 0 or step == config.steps - 1:
        # print(f"{step=}", loss.item())

        with torch.no_grad():
            task_ids, task_bits, parity = dataset.get_batch(batch_sz=config.eval_batch_sz)
            logits = model((task_ids, task_bits, ()))
            loss = F.cross_entropy(logits, parity)
            losses_overall.append(loss.item())
            print(f"{step=} val loss", loss.item())

        for task_index in range(config.n_control_bits):
            alpha = torch.tensor(0.0, requires_grad=True)
            # to check: is grad for the non-alpha param disabled?
            new_params = {name: old_params[name] + param_step[name] * alpha for name in old_params}

            task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
                batch_sz=config.eval_batch_sz, task_ids=torch.tensor(task_index)
            )
            logits = torch.func.functional_call(model, new_params, ((task_ids, task_bits, ()),))
            loss = F.cross_entropy(logits, parity)
            loss.backward()

            # print(task_index, alpha.grad)

            losses_by_task_for_step.append(loss.item())
            loss_gradients_by_task_for_step.append(alpha.grad)

        losses_by_task.append(losses_by_task_for_step)
        loss_gradients_by_task.append(loss_gradients_by_task_for_step)
losses_by_task = np.array(losses_by_task)
losses_overall = np.array(losses_overall)
loss_gradients_by_task = np.array(loss_gradients_by_task)

print(losses_by_task.shape)
x = np.arange(losses_by_task.shape[0]) * config.eval_interval
n_lines = losses_by_task.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    x,
    losses_by_task[:, :],
    label=list(np.arange(config.n_control_bits)),
    linewidth=0.5,
)

plt.plot(x, losses_overall, label="overall", color="red", linewidth=1)
plt.legend()
# plt.xscale("log")
plt.show()


n_lines = loss_gradients_by_task.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    x,
    loss_gradients_by_task[:, :],
    label=list(np.arange(config.n_control_bits)),
    linewidth=0.5,
)
plt.legend()
# plt.xscale("log")
plt.show()


# breakpoint()

# if step % config.eval_interval == 0 or step == config.steps - 1:
#     print(f"{step=}", loss.item())

#     losses_by_task_for_step = []
#     with torch.no_grad():
#         task_ids, task_bits, parity = dataset.get_batch(batch_sz=config.eval_batch_sz)
#         logits = model((task_ids, task_bits, ()))
#         val_loss = F.cross_entropy(logits, parity)
#         losses_by_task_for_step.append(val_loss.item())

#         for i in range(config.n_control_bits):
#             task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
#                 batch_sz=config.eval_batch_sz, task_ids=torch.tensor(i)
#             )
#             logits = model((task_ids, task_bits, ()))
#             loss = F.cross_entropy(logits, parity)
#             # print(i, loss.item())
#             losses_by_task_for_step.append(loss.item())

#     losses_by_step_and_task.append(losses_by_task_for_step)

# losses_by_step_and_task = np.array(losses_by_step_and_task)

# x = np.arange(losses_by_step_and_task.shape[0]) * config.eval_interval

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
