import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

color_by_log = True

n_components = 100
n = 10000
n_tasks = 5
n_bins = 100

# Dummy data: (n_components, n_tasks, n)
# ci_values_by_component_task = np.array(
#     [[F.sigmoid(torch.randn(n) * 4).numpy() for _ in range(n_tasks)] for _ in range(n_components)]
# )

ci_values_by_component_task = np.array(
    [[F.sigmoid(torch.randn(n) * 4).numpy() for _ in range(n_tasks)] for _ in range(n_components)]
)


def make_freqs(values_2d):
    """values_2d: (n_components, n_samples)"""
    freqs = []
    for ci_values in values_2d:
        counts, bin_edges = np.histogram(ci_values, bins=n_bins, range=(0, 1))
        freqs.append(counts / counts.sum())
    return np.array(freqs)


# All tasks combined: (n_components, n_tasks * n) -> freqs
all_freqs = make_freqs(ci_values_by_component_task.reshape(n_components, -1))
# Per task: list of (n_components, n_bins)
task_freqs = [make_freqs(ci_values_by_component_task[:, t, :]) for t in range(n_tasks)]

all_freqs_list = [all_freqs] + task_freqs
global_vmax = max(f.max() for f in all_freqs_list)
norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(vmin=1e-4, vmax=global_vmax)

titles = ["All Tasks"] + [f"Task {t}" for t in range(n_tasks)]
# fig, axes = plt.subplots(1, n_tasks + 1, figsize=(4 * (n_tasks + 1), 6), sharey=True)

# for ax, freqs, title in zip(axes, all_freqs_list, titles):
#     ax.imshow(
#         freqs,
#         aspect="auto",
#         cmap="viridis",
#         norm=norm,
#         extent=[0, 1, n_components, 0],
#     )
#     ax.set_xlabel("CI Value")
#     ax.set_title(title)

# axes[0].set_ylabel("Component")
# fig.colorbar(
#     plt.cm.ScalarMappable(cmap="viridis", norm=norm),
#     ax=axes.tolist(),
#     label=f"Frequency ({'log' if color_by_log else 'linear'} scale)",
# )
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1, n_tasks + 1, figsize=(4 * (n_tasks + 1), 6), sharey=True)

# for ax, freqs, title in zip(axes, all_freqs_list, titles):
#     im = ax.imshow(
#         freqs,
#         aspect="auto",
#         cmap="viridis",
#         norm=norm,
#         extent=[0, 1, n_components, 0],
#     )
#     ax.set_xlabel("CI Value")
#     ax.set_title(title)

# axes[0].set_ylabel("Component")
# fig.colorbar(
#     im,
#     ax=axes.tolist(),
#     label=f"Frequency ({'log' if color_by_log else 'linear'} scale)",
#     # shrink=0.8,
#     pad=0.02,
# )
# plt.tight_layout()
# plt.show()


fig = plt.figure(figsize=(4 * (n_tasks + 1), 6))
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
        extent=[0, 1, n_components, 0],
    )
    ax.set_xlabel("CI Value")
    ax.set_title(title)

axes[0].set_ylabel("Component")
fig.colorbar(im, cax=cax, label=f"Frequency ({'log' if color_by_log else 'linear'} scale)")
plt.tight_layout()
plt.show()


# n_components = 100
# n = 1000

# n_tasks = 5


# ci_values_by_component = []
# for _ in range(n_components):
#     ci_values_pre_sigmoid = torch.randn(n) * 4
#     ci_values = F.sigmoid(ci_values_pre_sigmoid)
#     ci_values_by_component.append(ci_values)

# ci_values_by_component = np.array(ci_values_by_component)


# # bin_edges = np.linspace(0, 1, 101)
# # bin_indices = np.searchsorted(bin_edges[1:-1], ci_values_by_component)  # (n_components, n)
# # freqs_by_component = np.zeros((n_components, 100))
# # np.add.at(freqs_by_component, (np.arange(n_components)[:, None], bin_indices), 1)
# # freqs_by_component /= freqs_by_component.sum(axis=1, keepdims=True)


# ci_freqs_by_component = []
# for ci_values in ci_values_by_component:
#     counts, bin_edges = np.histogram(ci_values, bins=100, range=(0, 1))
#     freqs = counts / counts.sum()
#     ci_freqs_by_component.append(freqs)
# ci_freqs_by_component = np.array(ci_freqs_by_component)


# fig, ax = plt.subplots(figsize=(10, 6))
# norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(
#     vmin=1e-5, vmax=ci_freqs_by_component.max()
# )
# ax.imshow(
#     ci_freqs_by_component,
#     aspect="auto",
#     cmap="viridis",
#     norm=norm,
#     extent=[0, 1, n_components, 0],
# )
# ax.set_xlabel("CI Value")
# ax.set_ylabel("Component")
# # ax.set_yticks(np.arange(n_components) + 0.5, labels=np.arange(n_components))
# fig.colorbar(
#     plt.cm.ScalarMappable(cmap="viridis", norm=norm),
#     ax=ax,
#     # label="Frequency (log scale)",
#     label=f"Count ({'log' if color_by_log else 'linear'} scale)",
# )
# plt.tight_layout()
# plt.show()

# 1 / 0

# # counts, bin_edges = np.histogram(ci_values, bins=100, range=(0, 1))
# # freqs = counts / counts.sum()

# # fig, ax = plt.subplots(figsize=(10, 1))

# # print(bin_edges)


# # norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(
# #     # vmin=max(counts.min(), 1),
# #     # vmin=0,
# #     vmin=1e-5,
# #     vmax=freqs.max(),
# # )
# # ax.imshow(
# #     freqs[np.newaxis, :],
# #     aspect="auto",
# #     cmap="viridis",
# #     norm=norm,
# #     extent=[bin_edges[0], bin_edges[-1], 0, 1],
# # )
# # ax.set_yticks([])
# # ax.set_xlabel("CI Value")
# # fig.colorbar(
# #     plt.cm.ScalarMappable(cmap="viridis", norm=norm),
# #     ax=ax,
# #     label=f"Count ({'log' if color_by_log else 'linear'} scale)",
# #     orientation="vertical",
# # )
# # plt.tight_layout()
# # plt.show()
