import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from storer import Storer
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig

# np.geomspace(16, 512, 11).round().astype(int) array([ 16,  23,  32,  45,  64,  91, 128, 181, 256, 362, 512])
# d_mlp = 256
storer = Storer("/modal_volume/mtsp_results/metrics/")
config = TrainConfig(seed=0)

storer.add_datestamp_prefix()
storer.add_prefix(f"train_mtsp_model_uniform_task_distribution/v10/{config.cache_key()}")

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
storer.add_prefix(f"model/{config.steps}")
key = "model"
# assert storer.exists(key), (d_mlp, storer)
assert storer.exists(key), storer
model.load_state_dict(storer.read(key))
losses_by_step_and_task: np.ndarray = storer.read("losses_by_step_and_task")

torch.manual_seed(0)
task_ids, task_bits, parity = dataset.get_batch(batch_sz=1024)
logits = model((task_ids, task_bits, ()))
val_loss = F.cross_entropy(logits, parity)
print(val_loss)

losses_by_task = []
for i in range(config.n_control_bits):
    task_ids, task_bits, parity = dataset.get_batch_for_task_ids(
        batch_sz=1000, task_ids=torch.tensor(i)
    )
    logits = model((task_ids, task_bits, ()))
    loss = F.cross_entropy(logits, parity)
    losses_by_task.append(loss.item())
losses_by_task = torch.tensor(losses_by_task)
print((losses_by_task < torch.log(torch.tensor(1.5))).float().mean())


x = np.arange(losses_by_step_and_task.shape[0]) * 1000

n_lines = losses_by_step_and_task.shape[1] - 1
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
plt.plot(
    x,
    losses_by_step_and_task[:, 1:],
    label=list(np.arange(config.n_control_bits)),
    linewidth=0.5,
)

plt.plot(x, losses_by_step_and_task[:, 0], label="overall", color="red", linewidth=1)
plt.legend()
# plt.xscale("log")
plt.yscale("log")
plt.show()
