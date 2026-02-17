import json
import os
import tempfile

import torch
import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig
from wandb_utils import load_wandb_model_artifact
from spd.log import logger
from spd.configs import Config
from spd.models.batch_and_loss_fns import recon_loss_kl
from spd.simple_trainer import optimize
from spd.utils.general_utils import save_pre_run_info
from spd.utils.run_utils import ExecutionStamp
from spd.utils.wandb_utils import init_wandb


spd_config = Config.from_file("config2.yaml")
target_model_wandb_run_path = "mutate/multitask-sparse-parity/5rvs7hyq"
spd_batch_sz = 64


api = wandb.Api()
run = api.run(target_model_wandb_run_path)

with tempfile.TemporaryDirectory() as tmpdir:
    config_file = run.file("train_config.json")
    config_file.download(root=tmpdir, replace=True)
    with open(os.path.join(tmpdir, "train_config.json")) as f:
        target_config = TrainConfig(**json.load(f))

device = torch.device('cuda')

train_loader = DataLoader(
    MultitaskSparseParityDataset(
        n_control_bits=target_config.n_control_bits,
        n_task_bits=target_config.n_task_bits,
        n_xored_bits=target_config.n_xored_bits,
        task_distribution_decay_rate=target_config.task_distribution_decay_rate,
        batch_sz=spd_batch_sz,
        size=None,
        device=device,
    ),
    batch_size=None,
)
eval_loader = DataLoader(
    MultitaskSparseParityDataset(
        n_control_bits=target_config.n_control_bits,
        n_task_bits=target_config.n_task_bits,
        n_xored_bits=target_config.n_xored_bits,
        task_distribution_decay_rate=target_config.task_distribution_decay_rate,
        batch_sz=spd_batch_sz,
        size=None,
        device=device,
    ),
    batch_size=None,
)

artifact_path = load_wandb_model_artifact(target_model_wandb_run_path)
target_model = MultitaskSparseParityModel(
    n_control_bits=target_config.n_control_bits,
    n_task_bits=target_config.n_task_bits,
    d_mlp=target_config.d_mlp,
).to(device)
target_model.load_state_dict(torch.load(artifact_path))


execution_stamp = ExecutionStamp.create(run_type="spd", create_snapshot=False)
out_dir = execution_stamp.out_dir
logger.info(f"Run ID: {execution_stamp.run_id}")
logger.info(f"Output directory: {out_dir}")

if spd_config.wandb_project:
    init_wandb(
        config=spd_config,
        project=spd_config.wandb_project,
        run_id=execution_stamp.run_id,
        name=spd_config.wandb_run_name,
        tags=[],
    )

optimize(
    target_model=target_model,
    config=spd_config,
    device=device,
    train_loader=train_loader,
    eval_loader=eval_loader,
    n_eval_steps=1000,
    reconstruction_loss=recon_loss_kl,
    out_dir=out_dir,
)

save_pre_run_info(
    save_to_wandb=spd_config.wandb_project is not None,
    out_dir=out_dir,
    spd_config=spd_config,
    sweep_params=None,
    target_model=target_model,
    train_config=None,
    task_name=spd_config.task_config.task_name,
)  # save final_config.yaml
