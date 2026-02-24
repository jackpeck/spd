import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

color_by_log = True

n_components = 800
n = 100
n_tasks = 10
n_bins = 10
k = 200

# Dummy data: (n_components, n_tasks, n)
ci_values_by_component_task = np.array(
    [[F.sigmoid(torch.randn(n) * 4).numpy() for _ in range(n_tasks)] for _ in range(n_components)]
)

mean_ci_values = ci_values_by_component_task.mean(1).mean(1)
mask = torch.topk(torch.tensor(mean_ci_values), k).indices

ci_values_by_component_task = ci_values_by_component_task[mask]
n_components_shown = min(k, n_components)


def make_freqs(values_2d):
    """values_2d: (n_components, n_samples)"""
    freqs = []
    for ci_values in values_2d:
        counts, bin_edges = np.histogram(ci_values, bins=n_bins, range=(0, 1))
        freqs.append(counts / counts.sum())
    return np.array(freqs)


# All tasks combined: (n_components, n_tasks * n) -> freqs
all_freqs = make_freqs(ci_values_by_component_task.reshape(n_components_shown, -1))
# Per task: list of (n_components, n_bins)
task_freqs = [make_freqs(ci_values_by_component_task[:, t, :]) for t in range(n_tasks)]

all_freqs_list = [all_freqs] + task_freqs
global_vmax = max(f.max() for f in all_freqs_list)
norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(vmin=1e-4, vmax=global_vmax)

titles = ["All Tasks"] + [f"Task {t}" for t in range(n_tasks)]


fig = plt.figure(figsize=(2 * (n_tasks + 1) + 2, 6))
gs = fig.add_gridspec(1, n_tasks + 2, width_ratios=[1] * (n_tasks + 1) + [0.05])

axes = [fig.add_subplot(gs[0, 0])]
for i in range(1, n_tasks + 1):
    axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
cax = fig.add_subplot(gs[0, -1])

for ax, freqs, title in zip(axes, all_freqs_list, titles):
    im = ax.imshow(
        freqs,
        aspect="auto",
        cmap="viridis",
        norm=norm,
        extent=[0, 1, n_components_shown, 0],
    )
    ax.set_xlabel("CI Value")
    ax.set_title(title)

for ax in axes[1:]:
    ax.tick_params(labelleft=False)

for ax in axes:
    ax.set_xticks([0, 0.5, 1])

axes[0].set_ylabel("Component")
fig.colorbar(im, cax=cax, label=f"Frequency ({'log' if color_by_log else 'linear'} scale)")
plt.tight_layout()
fig.savefig("ci_plot.png", dpi=300, bbox_inches="tight")
plt.show()
