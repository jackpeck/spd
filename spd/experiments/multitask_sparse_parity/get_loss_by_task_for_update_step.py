import hashlib
import inspect
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
    steps: int = 500000
    lr: float = 1e-3
    n_control_bits: int = 15
    n_task_bits: int = 30
    n_xored_bits: int = 4
    task_distribution_decay_rate: float = 0
    batch_sz: int = 1024
    code_version: str = "v4"
    eval_batch_sz: int = 1024
    eval_interval: int = 1
    model_src: str = inspect.getsource(MultitaskSparseParityModel)

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
key = "model"
if not storer.exists(key):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_gradients_by_task = []
    losses_by_task = []
    losses_overall = []

    train_dataloader = DataLoader(dataset, batch_size=None)

    old_params = None
    for step, (task_ids, task_bits, parity) in enumerate(train_dataloader):
        optimizer.zero_grad()

        logits = model((task_ids, task_bits, ()))
        loss = F.cross_entropy(logits, parity)

        old_params = {name: p.detach().clone() for name, p in model.named_parameters()}

        loss.backward()
        optimizer.step()

        # param_step = {name: p - old_params[name] for name, p in model.named_parameters()}

        loss_gradients_by_task_for_step = []
        losses_by_task_for_step = []

        if step % config.eval_interval == 0 or step == config.steps - 1:
            # print(f"{step=}", loss.item())

            param_step = None
            # if old_params is not None:
            param_step = {
                name: p.detach().clone() - old_params[name] for name, p in model.named_parameters()
            }
            # old_params = {name: p.detach().clone() for name, p in model.named_parameters()}

            with torch.no_grad():
                task_ids, task_bits, parity = dataset.get_batch(batch_sz=config.eval_batch_sz)
                logits = model((task_ids, task_bits, ()))
                loss = F.cross_entropy(logits, parity)
                losses_overall.append(loss.item())
                if step % 100 == 0:
                    print(f"{step=} val loss", loss.item())

            if param_step is not None:  # note: list is missing first entry / is off by one
                for task_index in range(config.n_control_bits):
                    alpha = torch.tensor(0.0, requires_grad=True)
                    # to check: is grad for the non-alpha param disabled?
                    new_params = {
                        name: old_params[name] + param_step[name] * alpha for name in old_params
                    }

                    task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
                        batch_sz=config.eval_batch_sz, task_ids=torch.tensor(task_index)
                    )
                    logits = torch.func.functional_call(
                        model, new_params, ((task_ids, task_bits, ()),)
                    )
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

    storer.write(key, model.state_dict())
    storer.write("losses_by_task", losses_by_task)
    storer.write("losses_overall", losses_overall)
    storer.write("loss_gradients_by_task", loss_gradients_by_task)

losses_by_task: np.ndarray = storer.read("losses_by_task")
losses_overall: np.ndarray = storer.read("losses_overall")
loss_gradients_by_task: np.ndarray = storer.read("loss_gradients_by_task")

print(losses_by_task.shape)
plt.figure(dpi=150)
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
plt.savefig(f"plots/{config.cache_key()}_losses_by_task.png")
with open(f"plots/{config.cache_key()}_config.txt", "w") as f:
    f.write(json.dumps(asdict(config), sort_keys=True, default=int, indent=True))
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


bucket_size = 10
n_steps = loss_gradients_by_task.shape[0]
n_buckets = n_steps // bucket_size
truncated = loss_gradients_by_task[: n_buckets * bucket_size]
bucketed = truncated.reshape(n_buckets, bucket_size, -1).mean(axis=1)

n_lines = bucketed.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    np.arange(n_buckets) * bucket_size * config.eval_interval,
    bucketed,
    label=list(np.arange(config.n_control_bits)),
    linewidth=1,
)
plt.legend()
plt.show()


bucket_size = 100
n_steps = losses_by_task.shape[0]
n_buckets = n_steps // bucket_size
truncated = losses_by_task[: n_buckets * bucket_size]
bucketed = truncated.reshape(n_buckets, bucket_size, -1).mean(axis=1)

n_lines = bucketed.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    np.arange(n_buckets) * bucket_size * config.eval_interval,
    bucketed,
    label=list(np.arange(config.n_control_bits)),
    linewidth=1,
)
plt.legend()
plt.show()


bucket_size = 100
n_steps = loss_gradients_by_task.shape[0] * config.eval_interval
n_buckets = loss_gradients_by_task.shape[0] // bucket_size
truncated = loss_gradients_by_task[: n_buckets * bucket_size]
bucketed = truncated.reshape(n_buckets, bucket_size, -1).mean(axis=1)

n_lines = bucketed.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    np.arange(n_buckets) * bucket_size * config.eval_interval,
    bucketed.mean(-1),
    label=list(np.arange(config.n_control_bits)),
    linewidth=0.5,
)
plt.legend()
plt.show()

import pandas as pd

ema_halflife = 100
alpha = 1 - np.exp(-np.log(2) / ema_halflife)
ema = pd.DataFrame(loss_gradients_by_task).ewm(alpha=alpha).mean().values

n_lines = ema.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    (np.arange(ema.shape[0]) * config.eval_interval)[10:],
    ema[10:],
    label=list(np.arange(config.n_control_bits)),
    linewidth=1,
)
plt.legend()
plt.savefig(f"plots/{config.cache_key()}_loss_gradient_by_task_ema_halflife={ema_halflife}.png")
plt.legend()
plt.show()


plt.clf()
ema_halflife = 100
alpha = 1 - np.exp(-np.log(2) / ema_halflife)
ema = pd.DataFrame(loss_gradients_by_task.var(-1)).ewm(alpha=alpha).mean().values

n_lines = ema.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    (np.arange(ema.shape[0]) * config.eval_interval)[10:],
    ema[10:],
    label=list(np.arange(config.n_control_bits)),
    linewidth=1,
)

plt.savefig(
    f"plots/{config.cache_key()}_loss_gradient_by_task_var_over_tasks_ema_halflife={ema_halflife}.png"
)

plt.show()


plt.clf()
ema_halflife = 1
alpha = 1 - np.exp(-np.log(2) / ema_halflife)
ema = pd.DataFrame(loss_gradients_by_task).ewm(alpha=alpha).mean().values
n_lines = ema.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    (np.arange(ema.shape[0]) * config.eval_interval)[10:],
    ema[10:],
    label=list(np.arange(config.n_control_bits)),
    linewidth=1,
)
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
