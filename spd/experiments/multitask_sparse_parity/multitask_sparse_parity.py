import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset


class MultitaskSparseParityDataset(IterableDataset):
    def __init__(
        self,
        n_control_bits,
        n_task_bits,
        n_xored_bits,
        batch_sz,
        task_distribution_decay_rate,
        seed=0,
        size=None,
    ):
        super().__init__()
        self.batch_sz = batch_sz
        self.size = size

        self.n_control_bits = n_control_bits
        self.n_task_bits = n_task_bits
        self.n_xored_bits = n_xored_bits

        self.task_distribution_decay_rate = task_distribution_decay_rate

        rng = torch.Generator()
        rng.manual_seed(0)
        self.selected_bits_by_task = (
            torch.randn((self.n_control_bits, self.n_task_bits), generator=rng)
            .sort()
            .indices[:, : self.n_xored_bits]
        )

    def get_batch(self, batch_sz):
        probs = F.normalize(
            (torch.arange(self.n_control_bits, dtype=torch.float) + 1)
            ** -(self.task_distribution_decay_rate),
            p=1,
            dim=0,
        )
        task_ids = torch.multinomial(probs, batch_sz, replacement=True)

        return self.get_batch_for_task_ids(batch_sz, task_ids)

    def get_batch_for_task_ids(self, batch_sz, task_ids):
        if len(task_ids.shape) == 0:
            task_ids = task_ids.repeat(batch_sz)
        task_bits = torch.randint(2, (batch_sz, self.n_task_bits))
        selected_bits_indexes = self.selected_bits_by_task[task_ids]
        selected_bits = task_bits.gather(1, selected_bits_indexes)
        parity = selected_bits.sum(-1) & 1
        return task_ids, task_bits, parity

    def __iter__(self):
        count = 0
        while self.size is None or count < self.size:
            yield self.get_batch(self.batch_sz)
            count += 1


class MultitaskSparseParityModel(nn.Module):
    def __init__(self, n_control_bits, n_task_bits, d_mlp):
        self.n_control_bits = n_control_bits
        self.n_task_bits = n_task_bits
        super().__init__()
        self.l1 = nn.Linear(n_control_bits + n_task_bits, d_mlp)
        # self.lb1 = nn.Linear(d_mlp, d_mlp)
        self.l2 = nn.Linear(d_mlp, 2)
        self.d_mlp = d_mlp

    def forward(self, batch):
        task_ids, task_bits, targets = batch
        x = torch.cat([F.one_hot(task_ids, self.n_control_bits), task_bits], dim=1).float()
        x = self.l1(x)
        # x = F.relu(x)
        # x = self.lb1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
