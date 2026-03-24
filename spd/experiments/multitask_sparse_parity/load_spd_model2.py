import json
import os
import tempfile

import einops
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig
from wandb_utils import load_wandb_model_artifact

from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.models.components import make_mask_infos
from spd.utils.module_utils import expand_module_patterns

# "wandb:mutate/spd/runs/s-387b3d65"
# "mutate/multitask-sparse-parity/runs/oniknm0t"  # d_mlp=14, n_control_bits=2, parity=2


def load_spd_model(target_model_path: str, spd_model_path: str):
    run_info = SPDRunInfo.from_path(spd_model_path)
    spd_config = run_info.config

    target_model_wandb_run_path = target_model_path

    api = wandb.Api()
    run = api.run(target_model_wandb_run_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = run.file("train_config.json")
        config_file.download(root=tmpdir, replace=True)
        with open(os.path.join(tmpdir, "train_config.json")) as f:
            target_config = TrainConfig(**json.load(f))

    target_model = MultitaskSparseParityModel(
        n_control_bits=target_config.n_control_bits,
        n_task_bits=target_config.n_task_bits,
        d_mlp=target_config.d_mlp,
    )
    # target model weights are overwritten in model.load_state_dict(comp_model_weights) so don't need to load here
    target_model.eval()
    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, spd_config.all_module_info)

    model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_fn_hidden_dims=spd_config.ci_fn_hidden_dims,
        ci_fn_type=spd_config.ci_fn_type,
        sigmoid_type=spd_config.sigmoid_type,
    )

    # print(run_info.checkpoint_path)
    comp_model_weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
    handle_deprecated_state_dict_keys_(comp_model_weights)
    model.load_state_dict(comp_model_weights)

    return target_config, spd_config, model


# print(model)

# dataloader = DataLoader(
#     MultitaskSparseParityDataset(
#         n_control_bits=target_config.n_control_bits,
#         n_task_bits=target_config.n_task_bits,
#         n_xored_bits=target_config.n_xored_bits,
#         task_distribution_decay_rate=target_config.task_distribution_decay_rate,
#         batch_sz=64,
#         size=None,
#     ),
#     batch_size=None,
# )
# batch = next(iter(dataloader))
# task_ids, task_bits, targets = batch

# loss = F.cross_entropy(model(batch), targets)
# print(loss)


# out = model(batch, cache_type="input")
# print(out.output[:5, :])
# loss = F.cross_entropy(out.output, targets)
# print(loss)


# ci_dict = model.calc_causal_importances(
#     pre_weight_acts=out.cache,
#     detach_inputs=True,
#     sampling=config.sampling,
# ).lower_leaky

# # print(ci_dict)

# print(task_ids[0], task_bits[0], targets[0])

# print(ci_dict["l1"].shape)

# layer_names = list(model.target_module_paths)
# ci = torch.cat([ci_dict[layer] for layer in layer_names], dim=1)

# print(ci.shape)

# component_ci_threshold = 0.9
# mask_infos = make_mask_infos(
#     # component_masks={k: torch.ones_like(v) for k, v in ci_dict.items()},
#     # component_masks={k: v for k, v in ci_dict.items()},
#     component_masks={k: (v > component_ci_threshold).float() for k, v in ci_dict.items()},
#     routing_masks="all",
# )

# logits_using_components = model(batch, cache_type="input", mask_infos=mask_infos).output
# loss_using_components = loss = F.cross_entropy(logits_using_components, targets)
# print(f"{loss_using_components=}")

# # loss_by_token_using_components = F.cross_entropy(
# #     einops.rearrange(logits_using_components[:, :-1], "b seq vocab -> b vocab seq"),
# #     batch[:, 1:],
# #     reduction="none",
# # )
# # print(f"mean loss using components = {loss_by_token_using_components.mean().item()}")


# for k, v in ci_dict.items():
#     print(k, (v > component_ci_threshold).float().mean())


# components = model.components["l1"]
# einops.einsum(components.V[:, 0], components.U[0], "d_in, d_out -> d_out d_in")

# idx = 0
# sns.heatmap(
#     einops.einsum(components.V[:, idx], components.U[idx], "d_in, d_out -> d_out d_in").detach(),
#     cmap="RdBu",
#     center=0,
# )
# plt.show()
