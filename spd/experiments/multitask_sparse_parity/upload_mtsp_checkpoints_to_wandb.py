# clauded

"""Upload training checkpoints from a TrainConfig run to wandb as separate runs."""

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from multitask_sparse_parity import MultitaskSparseParityModel
from storer import Storer
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig


@dataclass
class UploadCheckpointsConfig:
    upload_every_n_steps: int = 10_000


def get_steps_to_upload(
    train_config: TrainConfig, upload_config: UploadCheckpointsConfig
) -> list[int]:
    assert upload_config.upload_every_n_steps % train_config.eval_steps == 0, (
        f"upload_every_n_steps ({upload_config.upload_every_n_steps}) must be a multiple of "
        f"eval_steps ({train_config.eval_steps})"
    )
    steps = set(range(0, train_config.steps, upload_config.upload_every_n_steps))
    steps.add(train_config.steps)  # always include final
    return sorted(steps)


def upload_checkpoints_to_wandb(
    train_config: TrainConfig,
    upload_config: UploadCheckpointsConfig,
):
    cache_key = train_config.cache_key()
    steps_to_upload = get_steps_to_upload(train_config, upload_config)

    storer = Storer("/modal_volume/mtsp_results/metrics/")
    storer.add_datestamp_prefix()
    storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{cache_key}")
    storer.add_prefix(f"model/{train_config.steps}")

    assert storer.exists("model"), f"Final model not found at {storer}"

    losses_by_step_and_task: np.ndarray = storer.read("losses_by_step_and_task")

    model = MultitaskSparseParityModel(
        d_mlp=train_config.d_mlp,
        n_control_bits=train_config.n_control_bits,
        n_task_bits=train_config.n_task_bits,
    )

    for step in steps_to_upload:
        is_final = step == train_config.steps
        checkpoint_key = "model" if is_final else f"step={step}/model"

        if not storer.exists(checkpoint_key):
            print(f"Checkpoint at step {step} not found, skipping")
            continue

        model.load_state_dict(storer.read(checkpoint_key))

        run_name = f"{cache_key}_step{step}"
        wandb.init(project="multitask-sparse-parity", name=run_name, id=run_name)
        wandb.config.update(dataclasses.asdict(train_config))
        wandb.config.update({"checkpoint_step": step, "is_final_checkpoint": is_final})

        with tempfile.TemporaryDirectory() as tmpdir:
            train_config_path = os.path.join(tmpdir, "train_config.json")
            with open(train_config_path, "w") as f:
                json.dump(dataclasses.asdict(train_config), f, indent=4, default=int)
            wandb.save(train_config_path, base_path=tmpdir)

            model_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact(f"mtsp-checkpoint-step{step}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

        # Log loss at this checkpoint step
        if is_final:
            eval_idx = len(losses_by_step_and_task) - 1
        else:
            eval_idx = step // train_config.eval_steps
        if eval_idx < len(losses_by_step_and_task):
            log_dict = {
                "step": step,
                "loss/overall": losses_by_step_and_task[eval_idx, 0],
            }
            for task_id in range(train_config.n_control_bits):
                log_dict[f"loss/task_{task_id}"] = losses_by_step_and_task[eval_idx, task_id + 1]
            wandb.log(log_dict)

        wandb.finish()
        print(f"Uploaded step {step} as '{run_name}'")


if __name__ == "__main__":
    train_config = TrainConfig()
    upload_config = UploadCheckpointsConfig()
    upload_checkpoints_to_wandb(train_config, upload_config)
