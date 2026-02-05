import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

n_control_bits = 5
n_task_bits = 10
n_xored_bits = 4
batch_sz = 64

torch.manual_seed(3)


selected_bits_by_task = torch.randn((n_control_bits, n_task_bits)).sort().indices[:, :n_xored_bits]
# print("selected_bits_by_task", selected_bits_by_task)


def get_batch(batch_sz):
    probs = F.normalize(
        (torch.arange(n_control_bits, dtype=torch.float) + 1) ** -(0.2 + 1), p=1, dim=0
    )
    task_ids = torch.multinomial(probs, batch_sz, replacement=True)

    return get_batch_for_task_ids(batch_sz, task_ids)


def get_batch_for_task_ids(batch_sz, task_ids):
    if len(task_ids.shape) == 0:
        task_ids = task_ids.repeat(batch_sz)
    task_bits = torch.randint(2, (batch_sz, n_task_bits))
    selected_bits_indexes = selected_bits_by_task[task_ids]
    selected_bits = task_bits.gather(1, selected_bits_indexes)
    parity = selected_bits.sum(-1) & 1
    return task_ids, task_bits, parity


class MultitaskSparseParityDataset(IterableDataset):
    def __init__(self, batch_sz, size=None):
        super().__init__()
        self.batch_sz = batch_sz
        self.size = size

    def __iter__(self):
        count = 0
        while self.size is None or count < self.size:
            yield get_batch(self.batch_sz)
            count += 1


class MultitaskSparseParityModel(nn.Module):
    def __init__(self, d_mlp=200):
        super().__init__()
        self.l1 = nn.Linear(n_control_bits + n_task_bits, d_mlp)
        self.l2 = nn.Linear(d_mlp, 2)

    def forward(self, batch):
        task_ids, task_bits, targets = batch
        x = torch.cat([F.one_hot(task_ids, n_control_bits), task_bits], dim=1).float()
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


# model = Model()

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# steps = 100000

# losses_by_step_and_task = []


# dataset = MultitaskSparseParityDataset(batch_sz=batch_sz, size=steps)
# dataloader = DataLoader(dataset, batch_size=None)

# for step, (task_ids, task_bits, parity) in enumerate(dataloader):
#     optimizer.zero_grad()

#     logits = model(task_ids, task_bits)
#     loss = F.cross_entropy(logits, parity)

#     loss.backward()
#     optimizer.step()
#     if step % 100 == 0 or step == steps - 1:
#         print(f"{step=}", loss.item())

#         losses_by_task_for_step = []
#         with torch.no_grad():
#             task_ids, task_bits, parity = get_batch(batch_sz=64)
#             logits = model(task_ids, task_bits)
#             loss = F.cross_entropy(logits, parity)
#             # print(i, loss.item())
#             losses_by_task_for_step.append(loss.item())

#             for i in range(n_control_bits):
#                 task_ids, task_bits, parity = get_batch_for_task_ids(
#                     batch_sz=64, task_ids=torch.tensor(i)
#                 )
#                 logits = model(task_ids, task_bits)
#                 loss = F.cross_entropy(logits, parity)
#                 # print(i, loss.item())
#                 losses_by_task_for_step.append(loss.item())

#         losses_by_step_and_task.append(losses_by_task_for_step)

# losses_by_step_and_task = np.array(losses_by_step_and_task)

# # plt.plot(
# #     np.arange(losses_by_step_and_task.shape[0]) * 100,
# #     losses_by_step_and_task,
# #     label=["overall", *np.arange(n_control_bits)],
# #     cmap="virdis",
# # )
# # plt.legend()
# # plt.xscale("log")
# # plt.show()


# # n_lines = losses_by_step_and_task.shape[1]
# # plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)))
# # plt.plot(
# #     np.arange(losses_by_step_and_task.shape[0]) * 100,
# #     losses_by_step_and_task,
# #     label=["overall", *np.arange(n_control_bits)],
# # )
# # plt.legend()
# # plt.xscale("log")
# # plt.show()


# x = np.arange(losses_by_step_and_task.shape[0]) * 100

# n_lines = losses_by_step_and_task.shape[1] - 1
# # plt.figure(dpi=300)
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
