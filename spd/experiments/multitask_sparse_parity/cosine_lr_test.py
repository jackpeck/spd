import matplotlib.pyplot as plt
import torch
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW([torch.tensor(1.0, requires_grad=True)])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=10000
)


lrs = []
for _ in range(10000):
    scheduler.step()
    lrs.append(scheduler.get_lr())

plt.plot(lrs)
plt.show()
