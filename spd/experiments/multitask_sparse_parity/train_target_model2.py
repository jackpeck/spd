import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from torch.utils.data import DataLoader

n_control_bits = 10
n_task_bits = 10
n_xored_bits = 4
batch_sz = 64

storer = Storer()


for d_mlp in [10, 30, 100, 300, 1000, 3000]:
    # for d_mlp in [20, 70]:
    for seed in range(3):
        key = f"20260206/multitask_sparse_parity/parameter_scaling/v1/d_mlp={d_mlp}/seed={seed}/val_loss"
        if storer.exists(key):
            print(f"{key=} exists, skipping")
            continue

        torch.manual_seed(seed)
        model = MultitaskSparseParityModel(
            d_mlp=d_mlp, n_control_bits=n_control_bits, n_task_bits=n_task_bits
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_sz = 64
        steps = 50000

        losses_by_step_and_task = []

        dataset = MultitaskSparseParityDataset(
            n_control_bits=n_control_bits,
            n_task_bits=n_task_bits,
            n_xored_bits=n_xored_bits,
            batch_sz=batch_sz,
            size=steps,
        )
        train_dataloader = DataLoader(dataset, batch_size=None)

        for step, (task_ids, task_bits, parity) in enumerate(train_dataloader):
            optimizer.zero_grad()

            logits = model((task_ids, task_bits, ()))
            loss = F.cross_entropy(logits, parity)

            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == steps - 1:
                print(f"{step=}", loss.item())

                losses_by_task_for_step = []
                with torch.no_grad():
                    task_ids, task_bits, parity = dataset.get_batch(batch_sz=64)
                    logits = model((task_ids, task_bits, ()))
                    val_loss = F.cross_entropy(logits, parity)
                    losses_by_task_for_step.append(val_loss.item())

                    for i in range(n_control_bits):
                        task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
                            batch_sz=64, task_ids=torch.tensor(i)
                        )
                        logits = model((task_ids, task_bits, ()))
                        loss = F.cross_entropy(logits, parity)
                        # print(i, loss.item())
                        losses_by_task_for_step.append(loss.item())

                losses_by_step_and_task.append(losses_by_task_for_step)

        losses_by_step_and_task = np.array(losses_by_step_and_task)

        task_ids, task_bits, parity = dataset.get_batch(batch_sz=1024)
        logits = model((task_ids, task_bits, ()))
        val_loss = F.cross_entropy(logits, parity)
        print(val_loss)

        storer.write(key, val_loss)

# x = np.arange(losses_by_step_and_task.shape[0]) * 100

# n_lines = losses_by_step_and_task.shape[1] - 1
# plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
# plt.plot(
#     x,
#     losses_by_step_and_task[:, 1:],
#     label=list(np.arange(n_control_bits)),
#     linewidth=0.5,
# )

# plt.plot(x, losses_by_step_and_task[:, 0], label="overall", color="red", linewidth=1)
# plt.legend()
# plt.xscale("log")
# plt.show()


# wandb.init(project="multitask-sparse-parity")

# for step_idx, step in enumerate(range(0, steps, 100)):
#     log_dict = {"step": step, "loss/overall": losses_by_step_and_task[step_idx, 0]}
#     for task_id in range(n_control_bits):
#         log_dict[f"loss/task_{task_id}"] = losses_by_step_and_task[step_idx, task_id + 1]
#     wandb.log(log_dict)

# torch.save(model.state_dict(), "model.pt")
# artifact = wandb.Artifact("multitask-sparse-parity-model", type="model")
# artifact.add_file("model.pt")
# wandb.log_artifact(artifact)

# wandb.finish()
