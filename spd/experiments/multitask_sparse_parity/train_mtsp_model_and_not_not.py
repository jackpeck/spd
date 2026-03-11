# Train multitask sparse parity model with target: a AND (NOT b) AND (NOT c)
# i.e. target is 1 only when first selected bit is 1 and the other two are 0
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


class AndNotNotDataset(MultitaskSparseParityDataset):
    """Same as MultitaskSparseParityDataset but target is a AND (NOT b) AND (NOT c)
    instead of XOR parity."""

    def get_batch_for_task_ids(self, batch_sz, task_ids):
        if len(task_ids.shape) == 0:
            task_ids = task_ids.repeat(batch_sz)
        task_bits = torch.randint(2, (batch_sz, self.n_task_bits), device=self.device)
        selected_bits_indexes = self.selected_bits_by_task[task_ids]
        selected_bits = task_bits.gather(1, selected_bits_indexes)
        # a AND (NOT b) AND (NOT c): 1 only when first bit is 1 and rest are 0
        target = selected_bits[:, 0] * (1 - selected_bits[:, 1]) * (1 - selected_bits[:, 2])
        return task_ids, task_bits, target


image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "coolname", "safetensors", "packaging", "transformers")
    .add_local_python_source("multitask_sparse_parity")
    .add_local_python_source("storer")
    .add_local_python_source("modal_utils")
)
app = modal.App(
    "train-mtsp-and-not-not",
    image=image,
    volumes={"/mtsp_results": modal.Volume.from_name("mtsp_results", create_if_missing=True)},
)


@dataclass(frozen=True)
class TrainConfig:
    d_mlp: int = 14
    seed: int = 0
    steps: int = 500_000
    lr: float = 1e-3
    n_control_bits: int = 1
    n_task_bits: int = 30
    n_xored_bits: int = 3
    task_distribution_decay_rate: float = 0.4
    batch_sz: int = 1024
    code_version: str = "v1-and-not-not"
    eval_batch_sz: int = 1024
    weight_decay: float = 0.1
    norm_loss: float = 0.0
    model_src: str = inspect.getsource(MultitaskSparseParityModel)

    def cache_key(self):
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                default=int,
            ).encode()
        ).hexdigest()[:16]


@app.function(timeout=700)
def train_model(config):
    print(config)
    storer = Storer("/modal_volume/mtsp_results/metrics/")
    storer.add_datestamp_prefix()
    storer.add_prefix(f"train_mtsp_model_and_not_not/v1/{config.cache_key()}")
    print(storer)

    torch.manual_seed(config.seed)
    model = MultitaskSparseParityModel(
        d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
    )
    dataset = AndNotNotDataset(
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
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(config.steps * 0.01), num_training_steps=config.steps
        )

        losses_by_step_and_task = []

        train_dataloader = DataLoader(dataset, batch_size=None)

        for step, (task_ids, task_bits, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            logits = model((task_ids, task_bits, ()))
            norm_loss = sum(p.abs().pow(1.5).sum() for p in model.parameters()) * config.norm_loss
            ce_loss = F.cross_entropy(logits, target)
            loss = ce_loss + norm_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 1000 == 0 or step == config.steps - 1:
                print(
                    f"{step=}",
                    loss.item(),
                    f"{ce_loss.item()=} {norm_loss.item()=}",
                    f"l2norm={sum(p.abs().pow(2).sum() for p in model.parameters())}",
                )

                losses_by_task_for_step = []
                with torch.no_grad():
                    task_ids, task_bits, target = dataset.get_batch(batch_sz=config.eval_batch_sz)
                    logits = model((task_ids, task_bits, ()))
                    val_loss = F.cross_entropy(logits, target)
                    losses_by_task_for_step.append(val_loss.item())

                    for i in range(config.n_control_bits):
                        task_ids, task_bits, target = dataset.get_batch_for_task_ids(
                            batch_sz=config.eval_batch_sz, task_ids=torch.tensor(i)
                        )
                        logits = model((task_ids, task_bits, ()))
                        loss = F.cross_entropy(logits, target)
                        losses_by_task_for_step.append(loss.item())

                losses_by_step_and_task.append(losses_by_task_for_step)

        losses_by_step_and_task = np.array(losses_by_step_and_task)

        state = model.state_dict()
        storer.write(key, state)
        storer.write("losses_by_step_and_task", losses_by_step_and_task)

    model.load_state_dict(storer.read(key))
    losses_by_step_and_task = storer.read("losses_by_step_and_task")
    torch.manual_seed(0)
    task_ids, task_bits, target = dataset.get_batch(batch_sz=1024)
    logits = model((task_ids, task_bits, ()))
    val_loss = F.cross_entropy(logits, target)
    print(val_loss)


@app.local_entrypoint()
def main():
    seeds = list(range(5))

    d_mlps = np.geomspace(16, 512, 11).round().astype(int)
    configs = [TrainConfig(d_mlp=d_mlp, seed=seed) for d_mlp in d_mlps for seed in seeds]

    print(configs)
    for result in train_model.map(configs):
        pass


if __name__ == "__main__":
    config = TrainConfig()
    train_model.local(config)
