import itertools
import random
import time
from dataclasses import dataclass

import einops
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm
from jaxtyping import Float
from torch import Tensor

from spd.data import train_loader_and_tokenizer
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path


@dataclass
class Config:
    wandb_path: str
    batch_size: int
    n_batches: int | None


config = Config(wandb_path="goodfire/spd/runs/s-8dc8cf09", batch_size=1, n_batches=50)


_, _, run_id = parse_wandb_run_path(config.wandb_path)

run_info = SPDRunInfo.from_path(config.wandb_path)
device = get_device()
model = ComponentModel.from_run_info(run_info).to(device)
model.eval()

spd_config = run_info.config
train_loader, tokenizer = train_loader_and_tokenizer(spd_config, config.batch_size)

layer_names = list(model.target_module_paths)
vocab_size = tokenizer.vocab_size
assert isinstance(vocab_size, int)


train_iter = iter(train_loader)
batches_processed = 0
last_log_time = time.time()
batch_range = range(config.n_batches) if config.n_batches is not None else itertools.count()

loss_diffs = []

for batch_idx in tqdm.tqdm(batch_range, desc="Harvesting"):
    try:
        batch_data = extract_batch_data(next(train_iter))
    except StopIteration:
        logger.info(f"Dataset exhausted at batch {batch_idx}. Processing complete.")
        break

    batch = batch_data.to(device)
    with torch.no_grad():
        out = model(batch, cache_type="input")
        probs = torch.softmax(out.output, dim=-1)

    assert spd_config.sampling == "continuous"
    ci_dict = model.calc_causal_importances(
        pre_weight_acts=out.cache,
        detach_inputs=True,
        sampling=spd_config.sampling,
    ).lower_leaky

    ci: Float[Tensor, "B S n_comp"] = torch.cat([ci_dict[layer] for layer in layer_names], dim=2)

    logits = out.output
    loss_by_token = F.cross_entropy(
        einops.rearrange(logits[:, :-1], "b seq vocab -> b vocab seq"),
        batch[:, 1:],
        reduction="none",
    )

    print(f"mean loss = {loss_by_token.mean().item()}")

    # print(batch_data)

    # print(tokenizer.decode(batch_data[0]))

    component_ci_threshold = -1
    mask_infos = make_mask_infos(
        # component_masks={k: torch.ones_like(v) for k, v in ci_dict.items()},
        # component_masks={k: v for k, v in ci_dict.items()},
        component_masks={k: (v > component_ci_threshold).float() for k, v in ci_dict.items()},
        routing_masks="all",
    )

    logits_using_components = model(batch, cache_type="input", mask_infos=mask_infos).output
    loss_by_token_using_components = F.cross_entropy(
        einops.rearrange(logits_using_components[:, :-1], "b seq vocab -> b vocab seq"),
        batch[:, 1:],
        reduction="none",
    )
    print(f"mean loss using components = {loss_by_token_using_components.mean().item()}")

    logits_using_components = model(batch, cache_type="input", mask_infos=mask_infos).output
    # F.cross_entropy(
    #     einops.rearrange(logits_using_components[:, :-1], "b seq vocab -> b vocab seq"),
    #     batch[:, 1:],
    # )

    assert batch.shape[0] == 1, (
        "batch_sz not 1"
    )  # below code selects one component to deactivate across seq len and depth *and batch*. if multiple batches probably want one per batch

    module_key = random.choice(list(mask_infos.keys()))

    ci_for_module = ci_dict[module_key]

    indices = torch.nonzero(ci_for_module > 0.9)
    # indices = torch.nonzero(ci_for_module > -1)
    # indices = torch.nonzero(ci_for_module < 0.1)
    seq_len = ci_for_module.shape[1]
    # can't do loss on last token, so don't select components in last seq pos
    indices = indices[indices[:, 1] < seq_len - 1]

    for _ in range(50):
        to_deactivate_idx = indices[torch.randint(len(indices), ())]
        # print(f"deactivating component {to_deactivate_idx} [batch seq component_idx]")
        mask_infos = make_mask_infos(
            # component_masks={k: torch.ones_like(v) for k, v in ci_dict.items()},
            # component_masks={k: v for k, v in ci_dict.items()},
            component_masks={k: (v > component_ci_threshold).float() for k, v in ci_dict.items()},
            routing_masks="all",
        )
        mask_infos[module_key].component_mask[*to_deactivate_idx] = 0
        logits_using_components_with_deactivated = model(
            batch, cache_type="input", mask_infos=mask_infos
        ).output
        loss_by_token_using_components_deactivated = F.cross_entropy(
            einops.rearrange(
                logits_using_components_with_deactivated[:, :-1], "b seq vocab -> b vocab seq"
            ),
            batch[:, 1:],
            reduction="none",
        )
        # print(
        #     f"mean loss using components with deactivated = {loss_by_token_using_components_deactivated.mean().item()}"
        # )

        deactivated_component_seq_idx = to_deactivate_idx[1]

        # print(
        #     f"mean loss, at positions after deactivated component ([{deactivated_component_seq_idx}:]) for:"
        #     f"\nmodel {loss_by_token[0, deactivated_component_seq_idx:].mean().item()}"
        #     f"\nusing components {loss_by_token_using_components[0, deactivated_component_seq_idx:].mean().item()}"
        #     f"\nwith deactivated component {loss_by_token_using_components_deactivated[0, deactivated_component_seq_idx:].mean().item()}"
        # )

        loss_using_components_at_later_positions = loss_by_token_using_components[
            0, deactivated_component_seq_idx:
        ]  # .mean()
        loss_using_components_at_later_positions_with_deactivated = (
            loss_by_token_using_components_deactivated[
                0, deactivated_component_seq_idx:
            ]  # .#mean()
        )

        loss_diff = (
            loss_using_components_at_later_positions_with_deactivated
            - loss_using_components_at_later_positions
        ).max()
        loss_diffs.append(loss_diff.item())

loss_diffs = torch.tensor(loss_diffs)


# plt.clf()
# # plt.hist(loss_diffs.clamp(-0.1,0.1), bins=1000)
# plt.hist(
#     # loss_diffs.clamp(-0.05, 0.05),
#     loss_diffs.clamp(-99, 2),
#     # loss_diffs.clamp(-0.1, 0.3),
#     bins=1000,
#     weights=F.normalize(torch.ones_like(loss_diffs), p=1, dim=0),
# )
# plt.yscale("log")
# # plt.axvline(x=0, color="red", linestyle="--")
# plt.show()


plt.clf()
# plt.hist(loss_diffs.clamp(-0.1,0.1), bins=1000)
plt.hist(
    # loss_diffs.clamp(-0.05, 0.05),
    loss_diffs.clamp(-0.01, 2),
    # loss_diffs.clamp(-0.1, 0.3),
    bins=1000,
    weights=F.normalize(torch.ones_like(loss_diffs), p=1, dim=0),
)
plt.yscale("log")
# plt.axvline(x=0, color="red", linestyle="--")
plt.show()
