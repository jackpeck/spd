#!/bin/bash
#SBATCH --job-name=train-spd-mtsp
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

set -euo pipefail
cd /home/jpeck/workspace/scaling/spd-multitasksparseparity/spd
source .venv/bin/activate
cd spd/experiments/multitask_sparse_parity
python train_spd_model.py
