# task_mask = (((2**10 + 2**9 + 2**11 + 2**14) >> torch.arange(config.n_control_bits)) & 1).bool()
task_mask = (((2**10 + 2**14) >> torch.arange(config.n_control_bits)) & 1).bool()

plt.figure(dpi=150)
x = np.arange(losses_by_task.shape[0]) * config.eval_interval
n_lines = losses_by_task.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)[task_mask]))
plt.plot(
    x,
    # losses_by_task[:, 10:11],
    losses_by_task[:, task_mask],
    label=list(np.arange(config.n_control_bits)[task_mask]),
    linewidth=0.5,
)
plt.plot(x, losses_overall, label="overall", color="red", linewidth=1)
plt.legend()
plt.savefig(f"plots/{config.cache_key()}_losses_by_task_for_specific_tasks.png")
plt.show()


task_mask = (((2**10 + 2**14) >> torch.arange(config.n_control_bits)) & 1).bool()

plt.figure(dpi=150)
x = np.arange(loss_gradients_by_task.shape[0]) * config.eval_interval
n_lines = loss_gradients_by_task.shape[1]
plt.gca().set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, n_lines)[task_mask]))
plt.plot(
    x,
    loss_gradients_by_task[:, task_mask],
    label=list(np.arange(config.n_control_bits)[task_mask]),
    linewidth=0.5,
)
plt.legend()
plt.show()

import pandas as pd

task_mask = (((2**10 + 2**14) >> torch.arange(config.n_control_bits)) & 1).bool()

plt.figure(dpi=150)
x = np.arange(loss_gradients_by_task.shape[0]) * config.eval_interval
data = loss_gradients_by_task[:, task_mask]
n_lines = data.shape[1]
colors = plt.cm.viridis(np.linspace(0, 1, loss_gradients_by_task.shape[1])[task_mask])
plt.gca().set_prop_cycle(color=colors)

window = 1000
percentile_pairs = [
    (0, 100),
    (5, 95),
    (10, 90),
    (15, 85),
    (20, 80),
    (25, 75),
    (30, 70),
    (35, 65),
    (40, 60),
    (45, 55),
]

for i in range(n_lines):
    col = data[:, i].numpy() if hasattr(data[:, i], "numpy") else data[:, i]
    series = pd.Series(col)
    rolling = series.rolling(window, center=True, min_periods=1)
    mean = rolling.mean().values

    # Shaded percentile bands (outer to inner for proper layering)
    for lo_p, hi_p in percentile_pairs:
        lo = rolling.quantile(lo_p / 100).values
        hi = rolling.quantile(hi_p / 100).values
        alpha = 0.03 + 0.07 * (50 - lo_p) / 50  # more opaque for wider bands
        plt.fill_between(x, lo, hi, color=colors[i], alpha=alpha)

    plt.plot(
        x,
        mean,
        color=colors[i],
        label=str(np.arange(config.n_control_bits)[task_mask][i]),
        linewidth=1,
    )

plt.legend()
# plt.ylim(-0.0007, 0.0002)
plt.show()
