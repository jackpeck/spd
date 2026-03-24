# clauded

"""Submit parallel SLURM jobs to run SPD training on each uploaded checkpoint."""

import subprocess
import tempfile
from pathlib import Path

from train_mtsp_model_uniform_task_distribution_modal import TrainConfig
from upload_mtsp_checkpoints_to_wandb import UploadCheckpointsConfig, get_steps_to_upload

REPO_ROOT = "/home/jpeck/workspace/scaling/spd-multitasksparseparity/spd"
EXPERIMENT_DIR = f"{REPO_ROOT}/spd/experiments/multitask_sparse_parity"


def submit_spd_for_checkpoints(
    train_config: TrainConfig,
    upload_config: UploadCheckpointsConfig,
):
    cache_key = train_config.cache_key()
    steps = get_steps_to_upload(train_config, upload_config)

    print(f"{cache_key=}")
    print(f"Submitting {len(steps)} SPD jobs for steps: {steps}")

    for step in steps:
        wandb_run_path = f"mutate/multitask-sparse-parity/runs/{cache_key}_step{step}"
        run_name = f"spd-{cache_key[:8]}-step{step}"

        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

set -euo pipefail
cd {REPO_ROOT}
source .venv/bin/activate
cd {EXPERIMENT_DIR}
python train_spd_model.py --target-wandb-run {wandb_run_path} --run-name {run_name}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False) as f:
            f.write(sbatch_content)
            sbatch_path = f.name

        result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
        print(f"step {step}: {result.stdout.strip()} ({run_name})")
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")


if __name__ == "__main__":
    train_config = TrainConfig()
    upload_config = UploadCheckpointsConfig()
    submit_spd_for_checkpoints(train_config, upload_config)
