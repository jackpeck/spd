import torch
import wandb
from multitask_sparse_parity import MultitaskSparseParityModel

api = wandb.Api()
run = api.run("mutate/multitask-sparse-parity/yzq90nhu")

artifact = [artifact for artifact in run.logged_artifacts() if artifact.type == "model"][-1]
artifact_dir = artifact.download(root="/tmp/wandb_artifacts")
model = MultitaskSparseParityModel()
model.load_state_dict(torch.load(f"{artifact_dir}/model.pt"))
model.eval()

print(model)
