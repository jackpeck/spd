import hashlib
import inspect
import json
from dataclasses import asdict, dataclass

import modal
import numpy as np
import torch
import torch.nn.functional as F
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from torch.utils.data import DataLoader

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "coolname", "safetensors", "packaging")
    .add_local_python_source("multitask_sparse_parity")
    .add_local_python_source("storer")
    .add_local_python_source("modal_utils")
)
app = modal.App(
    "get-loss-by-task-parallel",
    image=image,
    volumes={"/mtsp_results": modal.Volume.from_name("mtsp_results", create_if_missing=True)},
)


@dataclass(frozen=True)
class TrainConfig:
    d_mlp: int = 64
    seed: int = 1
    steps: int = 500000
    lr: float = 1e-3
    n_control_bits: int = 15
    n_task_bits: int = 30
    n_xored_bits: int = 4
    task_distribution_decay_rate: float = 0
    batch_sz: int = 1024
    code_version: str = "v4"
    eval_batch_sz: int = 1024
    eval_interval: int = 1
    model_src: str = inspect.getsource(MultitaskSparseParityModel)

    def cache_key(self):
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                default=int,
            ).encode()
        ).hexdigest()[:16]


config = TrainConfig()
N_HOSTS = 500


def make_storer():
    storer = Storer("/modal_volume/mtsp_results/metrics/")
    storer.add_datestamp_prefix()
    storer.add_prefix(f"get_loss_by_task_for_update_step/v11/{config.cache_key()}")
    storer.add_prefix(f"model/{config.steps}")
    return storer


@app.function(timeout=3600)
def run_training_with_partial_eval(host_index: int, n_hosts: int):
    storer = make_storer()

    partial_storer = make_storer()
    partial_storer.add_prefix(f"partial/host_{host_index}")

    if partial_storer.exists("losses_by_task"):
        print(f"Host {host_index}: partial results already exist, skipping")
        return

    torch.manual_seed(config.seed)
    model = MultitaskSparseParityModel(
        d_mlp=config.d_mlp, n_control_bits=config.n_control_bits, n_task_bits=config.n_task_bits
    )
    dataset = MultitaskSparseParityDataset(
        n_control_bits=config.n_control_bits,
        n_task_bits=config.n_task_bits,
        n_xored_bits=config.n_xored_bits,
        task_distribution_decay_rate=config.task_distribution_decay_rate,
        batch_sz=config.batch_sz,
        size=config.steps,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train_dataloader = DataLoader(dataset, batch_size=None)

    loss_gradients_by_task = []
    losses_by_task = []
    losses_overall = []

    old_params = None
    eval_step_index = 0
    for step, (task_ids, task_bits, parity) in enumerate(train_dataloader):
        optimizer.zero_grad()
        logits = model((task_ids, task_bits, ()))
        loss = F.cross_entropy(logits, parity)
        old_params = {name: p.detach().clone() for name, p in model.named_parameters()}
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.steps - 1:
            my_eval = eval_step_index % n_hosts == host_index

            # Save RNG state before eval so all hosts stay in sync
            # (eval consumes RNG via dataset.get_batch calls)
            rng_state = torch.get_rng_state()

            if my_eval:
                param_step = {
                    name: p.detach().clone() - old_params[name]
                    for name, p in model.named_parameters()
                }

                with torch.no_grad():
                    task_ids_eval, task_bits_eval, parity_eval = dataset.get_batch(
                        batch_sz=config.eval_batch_sz
                    )
                    logits_eval = model((task_ids_eval, task_bits_eval, ()))
                    loss_eval = F.cross_entropy(logits_eval, parity_eval)
                    losses_overall.append(loss_eval.item())

                losses_by_task_for_step = []
                loss_gradients_by_task_for_step = []
                for task_index in range(config.n_control_bits):
                    alpha = torch.tensor(0.0, requires_grad=True)
                    new_params = {
                        name: old_params[name] + param_step[name] * alpha for name in old_params
                    }
                    task_ids_t, task_bits_t, parity_t = dataset.get_batch_for_task_ids(
                        batch_sz=config.eval_batch_sz, task_ids=torch.tensor(task_index)
                    )
                    logits_t = torch.func.functional_call(
                        model, new_params, ((task_ids_t, task_bits_t, ()),)
                    )
                    loss_t = F.cross_entropy(logits_t, parity_t)
                    loss_t.backward()
                    losses_by_task_for_step.append(loss_t.item())
                    loss_gradients_by_task_for_step.append(alpha.grad)

                losses_by_task.append(losses_by_task_for_step)
                loss_gradients_by_task.append(loss_gradients_by_task_for_step)

            if step % 1000 == 0:
                # print(f"Host {host_index}: {step=} val loss {loss_eval.item():.4f}")
                print(f"Host {host_index}: {step=}")

            # Restore RNG state so all hosts remain in sync for subsequent training steps
            torch.set_rng_state(rng_state)
            eval_step_index += 1

    if host_index == 0 and not storer.exists("model"):
        storer.write("model", model.state_dict())

    partial_storer.write("losses_by_task", np.array(losses_by_task))
    partial_storer.write("losses_overall", np.array(losses_overall))
    partial_storer.write("loss_gradients_by_task", np.array(loss_gradients_by_task))
    print(f"Host {host_index}: done, saved {len(losses_by_task)} eval steps")


@app.local_entrypoint()
def main():
    n_hosts = N_HOSTS
    storer = make_storer()

    try:
        if storer.exists("losses_by_task"):
            print("Collated results already exist, skipping")
            return
    except Exception:
        pass  # directory doesn't exist on Modal volume yet

    for result in run_training_with_partial_eval.starmap([(i, n_hosts) for i in range(n_hosts)]):
        pass

    # Collate partial results by interleaving
    partial_arrays = {}
    for i in range(n_hosts):
        ps = make_storer()
        ps.add_prefix(f"partial/host_{i}")
        partial_arrays[i] = {
            "losses_by_task": ps.read("losses_by_task"),
            "losses_overall": ps.read("losses_overall"),
            "loss_gradients_by_task": ps.read("loss_gradients_by_task"),
        }

    total_eval_steps = sum(len(partial_arrays[i]["losses_overall"]) for i in range(n_hosts))

    for key in ["losses_by_task", "losses_overall", "loss_gradients_by_task"]:
        parts = [partial_arrays[i][key] for i in range(n_hosts)]
        shape = list(parts[0].shape)
        shape[0] = total_eval_steps
        full = np.empty(shape)
        for i, part in enumerate(parts):
            full[i::n_hosts] = part
        storer.write(key, full)

    print(f"Collated {total_eval_steps} eval steps from {n_hosts} hosts")
