import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

color_by_log = True


n_components = 100
n = 1000

n_tasks = 5


ci_values_by_component = []
for _ in range(n_components):
    ci_values_pre_sigmoid = torch.randn(n) * 4
    ci_values = F.sigmoid(ci_values_pre_sigmoid)
    ci_values_by_component.append(ci_values)

ci_values_by_component = np.array(ci_values_by_component)


# bin_edges = np.linspace(0, 1, 101)
# bin_indices = np.searchsorted(bin_edges[1:-1], ci_values_by_component)  # (n_components, n)
# freqs_by_component = np.zeros((n_components, 100))
# np.add.at(freqs_by_component, (np.arange(n_components)[:, None], bin_indices), 1)
# freqs_by_component /= freqs_by_component.sum(axis=1, keepdims=True)


ci_freqs_by_component = []
for ci_values in ci_values_by_component:
    counts, bin_edges = np.histogram(ci_values, bins=100, range=(0, 1))
    freqs = counts / counts.sum()
    ci_freqs_by_component.append(freqs)
ci_freqs_by_component = np.array(ci_freqs_by_component)


fig, ax = plt.subplots(figsize=(10, 6))
norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(
    vmin=1e-5, vmax=ci_freqs_by_component.max()
)
ax.imshow(
    ci_freqs_by_component,
    aspect="auto",
    cmap="viridis",
    norm=norm,
    extent=[0, 1, n_components, 0],
)
ax.set_xlabel("CI Value")
ax.set_ylabel("Component")
# ax.set_yticks(np.arange(n_components) + 0.5, labels=np.arange(n_components))
fig.colorbar(
    plt.cm.ScalarMappable(cmap="viridis", norm=norm),
    ax=ax,
    # label="Frequency (log scale)",
    label=f"Count ({'log' if color_by_log else 'linear'} scale)",
)
plt.tight_layout()
plt.show()

1 / 0

# counts, bin_edges = np.histogram(ci_values, bins=100, range=(0, 1))
# freqs = counts / counts.sum()

# fig, ax = plt.subplots(figsize=(10, 1))

# print(bin_edges)


# norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(
#     # vmin=max(counts.min(), 1),
#     # vmin=0,
#     vmin=1e-5,
#     vmax=freqs.max(),
# )
# ax.imshow(
#     freqs[np.newaxis, :],
#     aspect="auto",
#     cmap="viridis",
#     norm=norm,
#     extent=[bin_edges[0], bin_edges[-1], 0, 1],
# )
# ax.set_yticks([])
# ax.set_xlabel("CI Value")
# fig.colorbar(
#     plt.cm.ScalarMappable(cmap="viridis", norm=norm),
#     ax=ax,
#     label=f"Count ({'log' if color_by_log else 'linear'} scale)",
#     orientation="vertical",
# )
# plt.tight_layout()
# plt.show()
