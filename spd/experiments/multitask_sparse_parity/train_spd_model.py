import torch
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from wandb_utils import load_wandb_model_artifact

from spd.configs import Config
from spd.models.batch_and_loss_fns import recon_loss_kl
from spd.simple_trainer import optimize
from spd.utils.general_utils import save_pre_run_info
from spd.utils.run_utils import ExecutionStamp

config = Config.from_file("config1.yaml")

batch_sz = 64
train_loader = DataLoader(
    MultitaskSparseParityDataset(batch_sz=batch_sz, size=100_000), batch_size=None
)
eval_loader = DataLoader(
    MultitaskSparseParityDataset(batch_sz=batch_sz, size=100_000), batch_size=None
)

target_model_wandb_run_path = "mutate/multitask-sparse-parity/1rvgs5j9"
artifact_path = load_wandb_model_artifact(target_model_wandb_run_path)
target_model = MultitaskSparseParityModel()
target_model.load_state_dict(torch.load(artifact_path))


out_dir = ExecutionStamp.create(run_type="spd", create_snapshot=False).out_dir

optimize(
    target_model=target_model,
    config=config,
    device="cpu",
    train_loader=train_loader,
    eval_loader=eval_loader,
    n_eval_steps=1000,
    reconstruction_loss=recon_loss_kl,
    out_dir=out_dir,
)

save_pre_run_info(
    save_to_wandb=config.wandb_project is not None,
    out_dir=out_dir,
    spd_config=config,
    sweep_params=None,
    target_model=target_model,
    train_config=None,
    task_name=config.task_config.task_name,
)  # save final_config.yaml
