# n_control_bits = 10
# n_task_bits = 30
# n_xored_bits = 4
# batch_sz = 64
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass

import modal

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "coolname", "safetensors", "packaging", "transformers")
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
    d_mlp: int = 32
    seed: int = 0
    steps: int = 100_000
    # steps: int = 10000
    # eval_steps: int = 500
    eval_steps: int = 1000
    lr: float = 1e-3
    n_control_bits: int = 2
    n_task_bits: int = 30
    n_xored_bits: int = 2
    task_distribution_decay_rate: float = 0.4
    batch_sz: int = 1024
    code_version: str = "v12"
    eval_batch_sz: int = 1024
    weight_decay: float = 0.1
    # norm_loss: float = 0.000001
    norm_loss: float = 0.0
    # save_model_weights_on_eval_step: bool = False
    save_model_weights_on_eval_step: bool = True
    model_src: str = inspect.getsource(MultitaskSparseParityModel)

    def cache_key(self):
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                default=int,  # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/66345356#comment133608565_72798532. default=int means if use np.int64 in config (e.g. from d_mlps = np.geomspace(32, 1024, 15).round().astype(int)) then hash is same as if use ints
            ).encode()
        ).hexdigest()[:16]


# for d_mlp in [32, 64, 128, 256, 512]:


@app.function(timeout=700)
def train_model(config):
    print(config)
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=config.steps, eta_min=1e-6
        # )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(config.steps * 0.01), num_training_steps=config.steps
        )

        losses_by_step_and_task = []

        train_dataloader = DataLoader(dataset, batch_size=None)

        for step, (task_ids, task_bits, parity) in enumerate(train_dataloader):
            optimizer.zero_grad()

            logits = model((task_ids, task_bits, ()))
            # lp_norm_loss = get_param_norm_to_p(model, p=2) * 0.0003 / 2
            norm_loss = sum(p.abs().pow(1.5).sum() for p in model.parameters()) * config.norm_loss
            ce_loss = F.cross_entropy(logits, parity)
            loss = ce_loss + norm_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % config.eval_steps == 0 or step == config.steps - 1:
                rng_state = torch.random.get_rng_state()
                if config.save_model_weights_on_eval_step:
                    state = model.state_dict()
                    storer.write(f"{step=}/model", state, overwrite_if_exists=True)

                print(
                    f"{step=}",
                    loss.item(),
                    f"{ce_loss.item()=} {norm_loss.item()=}",
                    f"l2norm={sum(p.abs().pow(2).sum() for p in model.parameters())}",
                )
                # print(f"{step=}", loss.item())

                losses_by_task_for_step = []
                with torch.no_grad():
                    task_ids, task_bits, parity = dataset.get_batch(batch_sz=config.eval_batch_sz)
                    logits = model((task_ids, task_bits, ()))
                    val_loss = F.cross_entropy(logits, parity)
                    losses_by_task_for_step.append(val_loss.item())

                    for i in range(config.n_control_bits):
                        task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
                            batch_sz=config.eval_batch_sz, task_ids=torch.tensor(i)
                        )
                        logits = model((task_ids, task_bits, ()))
                        loss = F.cross_entropy(logits, parity)
                        # print(i, loss.item())
                        losses_by_task_for_step.append(loss.item())

                losses_by_step_and_task.append(losses_by_task_for_step)
                torch.random.set_rng_state(rng_state)

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


# config = TrainConfig(n_xored_bits=10)
# train_model.local(config)


# train_model.local(TrainConfig())


@app.local_entrypoint()
def main():
    # config = TrainConfig(d_mlp=64)
    # train_model.remote(config)

    seeds = list(range(5))

    # d_mlps = [32, 64, 128, 256, 512]
    # d_mlps = np.geomspace(32, 2756, 19).round().astype(int)
    d_mlps = np.geomspace(16, 512, 11).round().astype(int)
    configs = [TrainConfig(d_mlp=d_mlp, seed=seed) for d_mlp in d_mlps for seed in seeds]

    # d_mlps = np.geomspace(16, 512, 11).round().astype(int)
    # n_control_bits_sweep = [2, 4, 6, 8, 10, 12, 14]
    # configs = [
    #     TrainConfig(d_mlp=512, seed=seed, n_control_bits=n_control_bits)
    #     for n_control_bits in n_control_bits_sweep
    #     for seed in seeds
    # ]
    print(configs)
    for result in train_model.map(configs):
        pass


if __name__ == "__main__":
    config = TrainConfig()
    train_model.local(config)

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
