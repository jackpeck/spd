import einops
import torch
import torch.nn.functional as F
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from wandb_utils import load_wandb_model_artifact

from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.models.components import make_mask_infos
from spd.utils.module_utils import expand_module_patterns

run_info = SPDRunInfo.from_path("/Users/jack/spd_out/spd/s-7a2de791/model_60000.pth")
config = run_info.config

target_model_wandb_run_path = "mutate/multitask-sparse-parity/1rvgs5j9"

artifact_path = load_wandb_model_artifact(target_model_wandb_run_path)
target_model = MultitaskSparseParityModel()
# target_model.load_state_dict(torch.load(artifact_path)) # target model weights are overwritten in model.load_state_dict(comp_model_weights)


target_model.eval()
target_model.requires_grad_(False)
module_path_info = expand_module_patterns(target_model, config.all_module_info)

model = ComponentModel(
    target_model=target_model,
    module_path_info=module_path_info,
    ci_fn_hidden_dims=config.ci_fn_hidden_dims,
    ci_fn_type=config.ci_fn_type,
    sigmoid_type=config.sigmoid_type,
)

print(run_info.checkpoint_path)
comp_model_weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
handle_deprecated_state_dict_keys_(comp_model_weights)
model.load_state_dict(comp_model_weights)


print(model)


dataloader = DataLoader(MultitaskSparseParityDataset(batch_sz=64), batch_size=None)
batch = next(iter(dataloader))
task_ids, task_bits, targets = batch

# loss = F.cross_entropy(model(batch), targets)
# print(loss)


out = model(batch, cache_type="input")
print(out.output[:5, :])
loss = F.cross_entropy(out.output, targets)
print(loss)


ci_dict = model.calc_causal_importances(
    pre_weight_acts=out.cache,
    detach_inputs=True,
    sampling=config.sampling,
).lower_leaky

# print(ci_dict)

print(task_ids[0], task_bits[0], targets[0])

print(ci_dict["l1"].shape)

layer_names = list(model.target_module_paths)
ci = torch.cat([ci_dict[layer] for layer in layer_names], dim=1)

print(ci.shape)

component_ci_threshold = -1
mask_infos = make_mask_infos(
    component_masks={k: torch.ones_like(v) for k, v in ci_dict.items()},
    # component_masks={k: v for k, v in ci_dict.items()},
    # component_masks={k: (v > component_ci_threshold).float() for k, v in ci_dict.items()},
    routing_masks="all",
)

logits_using_components = model(batch, cache_type="input", mask_infos=mask_infos).output
loss_using_components = loss = F.cross_entropy(logits_using_components, targets)
print(loss_using_components)

# loss_by_token_using_components = F.cross_entropy(
#     einops.rearrange(logits_using_components[:, :-1], "b seq vocab -> b vocab seq"),
#     batch[:, 1:],
#     reduction="none",
# )
# print(f"mean loss using components = {loss_by_token_using_components.mean().item()}")
