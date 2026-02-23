import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

color_by_log = False

n = 10000
ci_values_pre_sigmoid = torch.randn(n) * 4
ci_values = F.sigmoid(ci_values_pre_sigmoid)

# counts, bin_edges = np.histogram(ci_values, bins=100)
counts, bin_edges = np.histogram(ci_values, bins=100, range=(0, 1))
freqs = counts / counts.sum()

fig, ax = plt.subplots(figsize=(10, 1))

print(bin_edges)


norm = (mcolors.LogNorm if color_by_log else mcolors.Normalize)(
    # vmin=max(counts.min(), 1),
    # vmin=0,
    vmin=1e-5,
    vmax=freqs.max(),
)
ax.imshow(
    freqs[np.newaxis, :],
    aspect="auto",
    cmap="viridis",
    norm=norm,
    extent=[bin_edges[0], bin_edges[-1], 0, 1],
)
ax.set_yticks([])
ax.set_xlabel("CI Value")
fig.colorbar(
    plt.cm.ScalarMappable(cmap="viridis", norm=norm),
    ax=ax,
    label=f"Count ({'log' if color_by_log else 'linear'} scale)",
    orientation="vertical",
)
plt.tight_layout()
plt.show()
