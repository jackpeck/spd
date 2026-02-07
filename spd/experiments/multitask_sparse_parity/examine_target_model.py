import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from multitask_sparse_parity import (
    MultitaskSparseParityDataset,
    MultitaskSparseParityModel,
    n_control_bits,
    n_task_bits,
    selected_bits_by_task,
)
from torch.utils.data import DataLoader
from wandb_utils import load_wandb_model_artifact

torch.manual_seed(0)

batch_sz = 64
train_loader = DataLoader(
    MultitaskSparseParityDataset(batch_sz=batch_sz, size=100_000), batch_size=None
)

target_model_wandb_run_path = "mutate/multitask-sparse-parity/1rvgs5j9"
artifact_path = load_wandb_model_artifact(target_model_wandb_run_path)
target_model = MultitaskSparseParityModel()
target_model.load_state_dict(torch.load(artifact_path))


batch = next(iter(train_loader))
task_ids, task_bits, parity = batch

loss = F.cross_entropy(target_model(batch), parity)
print(loss)


# task_bits = torch.zeros((1, n_task_bits))
task_bits = torch.cartesian_prod(*[torch.tensor([0, 1]) for _ in range(n_task_bits)])

task_ids = torch.tensor([0]).repeat(task_bits.shape[0])
print(task_ids.shape)
# exit()
print(target_model((task_ids, task_bits, ())))

print(selected_bits_by_task)


x = torch.cat([F.one_hot(task_ids, n_control_bits), task_bits], dim=1).float()
x_post_l1 = target_model.l1(x)
x_post_relu = F.relu(x_post_l1)
x_post_l2 = target_model.l2(x_post_relu)

sns.heatmap(x_post_relu.detach())
plt.show()
