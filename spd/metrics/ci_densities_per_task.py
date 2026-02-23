from typing import Any, ClassVar, override

import torch
from jaxtyping import Int
from PIL import Image
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.plotting import plot_ci_densities_per_task
from spd.utils.distributed_utils import all_reduce


class CIDensitiesPerTask(Metric[Any, Any]):
    """Accumulates per-component, per-task CI histogram counts incrementally.

    Maintains integer bin counts rather than storing raw CI values, so memory
    usage is O(n_tasks * n_components * n_bins) regardless of how many batches
    are processed.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel[Any, Any],
        n_bins: int = 100,
        device: str = "cpu",
    ):
        assert (
            type(model.target_model).__name__ == "MultitaskSparseParityModel"
        )  # can't use isinstance(model.target_model), MultitaskSparseParityModel as import paths are different so this fails
        n_tasks: int = model.target_model.n_control_bits  # type: ignore[union-attr]
        self.n_tasks = n_tasks
        self.n_bins = n_bins
        self.batches_seen = 0

        # Per-task counts: (n_tasks, C, n_bins)
        # "All tasks" counts are just the sum over the task dim.
        self.counts: dict[str, Int[Tensor, "n_tasks C n_bins"]] = {
            module_name: torch.zeros(
                n_tasks, model.module_to_c[module_name], n_bins, dtype=torch.long, device=device
            )
            for module_name in model.components
        }

        bin_edges = torch.linspace(0, 1, n_bins + 1, device=device)
        self.bin_edges = bin_edges

    @override
    def update(self, *, batch: Any, ci: CIOutputs, **_: Any) -> None:
        self.batches_seen += 1

        task_ids: Int[Tensor, " batch"] = batch[0]

        for module_name, ci_vals in ci.lower_leaky.items():
            # ci_vals: (batch, C)
            # Clamp to [0, 1] to handle any minor numerical overshoot
            clamped = ci_vals.clamp(0.0, 1.0)
            # bucketize returns bin index for each value
            # right=False: bins are [edge_i, edge_i+1), last bin is [edge_{n-1}, edge_n]
            bin_indices = torch.bucketize(clamped, self.bin_edges[1:-1])
            # bin_indices: (batch, C), values in [0, n_bins-1]

            for task_id in range(self.n_tasks):
                mask = task_ids == task_id
                if not mask.any():
                    continue
                task_bins = bin_indices[mask]  # (n_task_samples, C)
                n_components = task_bins.shape[1]
                for c in range(n_components):
                    self.counts[module_name][task_id, c].scatter_add_(
                        0, task_bins[:, c], torch.ones_like(task_bins[:, c])
                    )

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.batches_seen > 0, "No batches seen yet"

        # Reduce counts across ranks
        reduced_counts: dict[str, Tensor] = {}
        for module_name, counts in self.counts.items():
            reduced_counts[module_name] = all_reduce(counts, op=ReduceOp.SUM)

        return plot_ci_densities_per_task(reduced_counts)
