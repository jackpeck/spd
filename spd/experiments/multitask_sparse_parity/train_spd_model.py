import torch
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader

from spd.configs import Config
from spd.models.batch_and_loss_fns import recon_loss_kl
from spd.simple_trainer import optimize
from spd.utils.run_utils import ExecutionStamp

config = Config.from_file("config1.yaml")

batch_sz = 64
train_loader = DataLoader(
    MultitaskSparseParityDataset(batch_sz=batch_sz, size=100_000), batch_size=None
)
eval_loader = DataLoader(
    MultitaskSparseParityDataset(batch_sz=batch_sz, size=100_000), batch_size=None
)

model = MultitaskSparseParityModel()


out_dir = ExecutionStamp.create(run_type="spd", create_snapshot=False).out_dir

# open(out_dir / "metrics.jsonl", "w").close()

optimize(
    target_model=model,
    config=config,
    device="cpu",
    train_loader=train_loader,
    eval_loader=eval_loader,
    n_eval_steps=1000,
    reconstruction_loss=recon_loss_kl,
    out_dir=out_dir,
)
