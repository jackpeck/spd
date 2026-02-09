import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from storer import Storer

storer = Storer()
storer.add_datestamp_prefix()

d_mlps = np.array([10, 20, 70, 30, 100, 300, 1000, 3000])
# d_mlps = np.array([20, 70])
points = []
for d_mlp in d_mlps:
    for seed in range(3):
        key = f"multitask_sparse_parity/parameter_scaling/v1/d_mlp={d_mlp}/seed={seed}/val_loss"
        points.append((d_mlp, storer.read(key)))

c_d_mlps, c_val_losses = zip(*points, strict=False)
df = pd.DataFrame(
    {
        "d_mlp": c_d_mlps,
        "val_loss": c_val_losses,
    }
)

# df2 = df.groupby("d_mlp")["val_loss"].std()
# plt.plot(df2.index, df2.values)
# plt.xlabel("d_mlp")
# plt.ylabel("val_loss")
# plt.yscale("log")
# plt.xscale("log")
# plt.show()


grouped = df.groupby("d_mlp")["val_loss"]
means = grouped.mean()
stds = grouped.std()

plt.errorbar(means.index, means.values, yerr=stds.values * 2, fmt="o-", capsize=4)
plt.xlabel("d_mlp")
plt.ylabel("val_loss")
plt.yscale("log")
plt.xscale("log")
plt.show()
