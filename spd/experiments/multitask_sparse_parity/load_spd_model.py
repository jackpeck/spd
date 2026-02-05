import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from wandb_utils import load_wandb_model_artifact

from spd.configs import Config
from spd.interfaces import RunInfo
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns


@dataclass
class SPDRunInfo(RunInfo[Config]):
    """Run info from training a ComponentModel (i.e. from an SPD run)."""

    config_class = Config
    config_filename = "final_config.yaml"
    checkpoint_prefix = "model"


config = SPDRunInfo.from_path("/Users/jack/spd_out/spd/s-dd7179ca/model_60000.pth").config

target_model_wandb_run_path = "mutate/multitask-sparse-parity/yzq90nhu"

artifact_path = load_wandb_model_artifact(target_model_wandb_run_path)
target_model = MultitaskSparseParityModel()
target_model.load_state_dict(torch.load(artifact_path))
target_model.eval()


target_model.eval()
target_model.requires_grad_(False)
module_path_info = expand_module_patterns(target_model, config.all_module_info)

component_model = ComponentModel(
    target_model=target_model,
    module_path_info=module_path_info,
    ci_fn_hidden_dims=config.ci_fn_hidden_dims,
    ci_fn_type=config.ci_fn_type,
    sigmoid_type=config.sigmoid_type,
)


print(component_model)


dataloader = DataLoader(MultitaskSparseParityDataset(batch_sz=64), batch_size=None)
batch = next(iter(dataloader))
targets = batch[2]

loss = F.cross_entropy(component_model(batch), targets)
print(loss)
