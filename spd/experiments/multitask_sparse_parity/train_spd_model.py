import json
import os
import subprocess
import tempfile


def get_least_loaded_gpu() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, mem = line.split(", ")
        gpus.append((int(idx), int(mem)))
    return min(gpus, key=lambda x: x[1])[0]


least_loaded_gpu = get_least_loaded_gpu()
print(f"Selected GPU {least_loaded_gpu} (least loaded by memory)")
os.environ["CUDA_VISIBLE_DEVICES"] = str(least_loaded_gpu)

import torch
import wandb
from generate_descriptive_run_name import generate_run_name
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig
from wandb_utils import load_wandb_model_artifact

from spd.configs import Config
from spd.log import logger
from spd.models.batch_and_loss_fns import recon_loss_kl
from spd.simple_trainer import optimize
from spd.utils.general_utils import save_pre_run_info
from spd.utils.run_utils import ExecutionStamp
from spd.utils.wandb_utils import init_wandb

spd_config = Config.from_file("config2.yaml")
run_name = spd_config.wandb_run_name or generate_run_name()
print(f"{run_name=}")

slurm_job_id = os.environ.get("SLURM_JOB_ID")
if slurm_job_id:
    subprocess.run(["scontrol", "update", f"JobId={slurm_job_id}", f"JobName={run_name}"])

# target_model_wandb_run_path = "mutate/multitask-sparse-parity/5rvs7hyq"

# seed=0
# n_control_bits=10
# a = """
# d_mlp=16 mutate/multitask-sparse-parity/runs/7xb3yyaf
# d_mlp=23 mutate/multitask-sparse-parity/runs/ycpc59g7
# d_mlp=32 mutate/multitask-sparse-parity/runs/gatrc939
# d_mlp=45 mutate/multitask-sparse-parity/runs/uvy85jm3
# d_mlp=64 mutate/multitask-sparse-parity/runs/oxw7olwc
# d_mlp=91 mutate/multitask-sparse-parity/runs/fdsbgws9
# d_mlp=128 mutate/multitask-sparse-parity/runs/zdmyecjh
# d_mlp=181 mutate/multitask-sparse-parity/runs/a51ve1e8
# d_mlp=256 mutate/multitask-sparse-parity/runs/c95cvma0
# d_mlp=362 mutate/multitask-sparse-parity/runs/shej8fad
# d_mlp=512 mutate/multitask-sparse-parity/runs/dcs8b6mq
# """
# a2 = [l.split(" ") for l in a.split("\n")[1:]]

# idx = 6
# d_mlp_str, target_model_wandb_run_path = a2[idx]


d_mlp_str, target_model_wandb_run_path = (
    "d_mlp=128",
    # "mutate/multitask-sparse-parity/runs/t1ndgltv",  # d_mlp=128, n_control_bits=5
    # "mutate/multitask-sparse-parity/runs/9h2kcm8r",  # d_mlp=128, n_control_bits=1
    "mutate/multitask-sparse-parity/runs/zdmyecjh",  # d_mlp=128, n_control_bits=10
)

print(f"{d_mlp_str=}, {target_model_wandb_run_path=}")

# target_model_wandb_run_path = "mutate/multitask-sparse-parity/5rvs7hyq"

api = wandb.Api()
run = api.run(target_model_wandb_run_path)

with tempfile.TemporaryDirectory() as tmpdir:
    config_file = run.file("train_config.json")
    config_file.download(root=tmpdir, replace=True)
    with open(os.path.join(tmpdir, "train_config.json")) as f:
        target_config = TrainConfig(**json.load(f))

device = torch.device("cuda")

train_loader = DataLoader(
    MultitaskSparseParityDataset(
        n_control_bits=target_config.n_control_bits,
        n_task_bits=target_config.n_task_bits,
        n_xored_bits=target_config.n_xored_bits,
        task_distribution_decay_rate=target_config.task_distribution_decay_rate,
        batch_sz=spd_config.batch_size,
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
        batch_sz=spd_config.eval_batch_size,
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
        name=run_name,
        tags=[],
    )

optimize(
    target_model=target_model,
    config=spd_config,
    device=device,
    train_loader=train_loader,
    eval_loader=eval_loader,
    n_eval_steps=spd_config.n_eval_steps,
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
