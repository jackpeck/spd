import json
import os
import tempfile
from dataclasses import dataclass

import einops
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from multitask_sparse_parity import MultitaskSparseParityDataset, MultitaskSparseParityModel
from torch.utils.data import DataLoader
from train_mtsp_model_uniform_task_distribution_modal import TrainConfig
from upload_mtsp_checkpoints_to_wandb import UploadCheckpointsConfig, get_steps_to_upload
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


@dataclass
class SPDCheckpointResult:
    spd_model_path: str
    model: ComponentModel


def _fetch_and_cache_checkpoint(
    step: int, cache_key: str, storer: "Storer"
) -> tuple[int, str, dict, "Config"] | None:
    """Fetch from wandb, cache state dict + metadata via Storer, return (step, path, state_dict, spd_config)."""
    from spd.configs import Config

    api = wandb.Api()
    run_name = f"spd-{cache_key[:8]}-step{step}"
    runs = list(api.runs("mutate/spd", filters={"display_name": run_name}))
    if not runs:
        print(f"No SPD run found for step {step} (searched for name={run_name}), skipping")
        return None
    assert len(runs) == 1, f"Expected 1 run for {run_name}, found {len(runs)}"

    spd_model_path = f"wandb:mutate/spd/runs/{runs[0].id}"
    target_model_path = f"mutate/multitask-sparse-parity/runs/{cache_key}_step{step}"

    _, spd_config, model = load_spd_model(target_model_path, spd_model_path)

    state_dict = model.state_dict()
    storer.write(f"step={step}/model", state_dict)
    cache_dir = storer.get_path(f"step={step}/model").parent
    (cache_dir / "spd_model_path.txt").write_text(spd_model_path)
    spd_config.to_file(cache_dir / "final_config.yaml")

    print(f"Fetched step {step}: {spd_model_path}")
    return step, spd_model_path, state_dict, spd_config


def _load_cached_checkpoint(
    step: int, storer: "Storer"
) -> tuple[int, str, dict, "Config"]:
    from spd.configs import Config

    state_dict = storer.read(f"step={step}/model")
    cache_dir = storer.get_path(f"step={step}/model").parent
    spd_model_path = (cache_dir / "spd_model_path.txt").read_text().strip()
    spd_config = Config.from_file(cache_dir / "final_config.yaml")
    print(f"Loaded step {step} from cache")
    return step, spd_model_path, state_dict, spd_config


def _build_component_model(train_config: TrainConfig, spd_config, state_dict: dict) -> ComponentModel:
    state_dict.pop("__dict__sentinel", None)
    target_model = MultitaskSparseParityModel(
        n_control_bits=train_config.n_control_bits,
        n_task_bits=train_config.n_task_bits,
        d_mlp=train_config.d_mlp,
    )
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
    handle_deprecated_state_dict_keys_(state_dict)
    model.load_state_dict(state_dict)
    return model


def load_spd_models_for_checkpoints(
    train_config: TrainConfig,
    upload_config: UploadCheckpointsConfig,
) -> dict[int, SPDCheckpointResult]:
    from concurrent.futures import ThreadPoolExecutor

    from storer import Storer

    cache_key = train_config.cache_key()
    steps = get_steps_to_upload(train_config, upload_config)
    storer = Storer(f"./spd_checkpoint_cache/{cache_key}")

    cached_steps = [s for s in steps if storer.exists(f"step={s}/model")]
    uncached_steps = [s for s in steps if s not in cached_steps]

    loaded: list[tuple[int, str, dict, object] | None] = [
        _load_cached_checkpoint(s, storer) for s in cached_steps
    ]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_fetch_and_cache_checkpoint, s, cache_key, storer)
            for s in uncached_steps
        ]
    loaded.extend(f.result() for f in futures)

    results: dict[int, SPDCheckpointResult] = {}
    for result in loaded:
        if result is None:
            continue
        step, spd_model_path, state_dict, spd_config = result
        model = _build_component_model(train_config, spd_config, state_dict)
        results[step] = SPDCheckpointResult(spd_model_path=spd_model_path, model=model)

    return results


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
