import dataclasses
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig

# d_mlps = np.geomspace(16, 512, 11).round().astype(int)
# seeds = list(range(5))

# configs = [TrainConfig(d_mlp=d_mlp, seed=seed) for d_mlp in d_mlps for seed in seeds]

# n_control_bits_sweep = [2, 4, 6, 8, 10, 12, 14]
# np.geomspace(16, 512, 11).round().astype(int) array([ 16,  23,  32,  45,  64,  91, 128, 181, 256, 362, 512])

# config = TrainConfig(d_mlp=512, seed=0, n_control_bits=10)
config = TrainConfig()
print(config)
storer = Storer("/modal_volume/mtsp_results/metrics/")
storer.add_datestamp_prefix()
storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{config.cache_key()}")
print(storer)

torch.manual_seed(config.seed)
model = MultitaskSparseParityModel(
    d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
)
storer.add_prefix(f"model/{config.steps}")
key = "model"
assert storer.exists(key)


model.load_state_dict(storer.read(key))
losses_by_step_and_task = storer.read("losses_by_step_and_task")
torch.manual_seed(0)


wandb.init(project="multitask-sparse-parity")
wandb.config.update(dataclasses.asdict(config))
with tempfile.TemporaryDirectory() as tmpdir:
    artifact = wandb.Artifact("multitask-sparse-parity-model", type="model")
    train_config_path = os.path.join(tmpdir, "train_config.json")
    model_path = os.path.join(tmpdir, "model.pt")
    with open(train_config_path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4, default=int)
    wandb.save(train_config_path, base_path=tmpdir)

    storer_path = os.path.join(tmpdir, "storer_path.txt")
    with open(storer_path, "w") as f:
        f.write(str(storer))
    wandb.save(storer_path, base_path=tmpdir)

    cache_key_path = os.path.join(tmpdir, "config.cache_key().txt")
    with open(cache_key_path, "w") as f:
        f.write(config.cache_key())
    wandb.save(cache_key_path, base_path=tmpdir)

    for step_idx, step in enumerate(range(0, config.steps, 1000)):
        log_dict = {"step": step, "loss/overall": losses_by_step_and_task[step_idx, 0]}
        for task_id in range(config.n_control_bits):
            log_dict[f"loss/task_{task_id}"] = losses_by_step_and_task[step_idx, task_id + 1]
        wandb.log(log_dict)

    torch.save(model.state_dict(), model_path)

    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
wandb.finish()
