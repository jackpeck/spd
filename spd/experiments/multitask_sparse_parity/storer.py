from pathlib import Path

import numpy as np
import safetensors.torch
import torch

TORCH_PREFIX = "torch:"
NP_PREFIX = "np:"


class Storer:
    def __init__(self, root_path=Path("./metrics/")):
        self.root_path = root_path

    def get_path(self, key):
        return self.root_path / (key + ".pt")

    def write(self, key, value, overwrite_if_exists=False):
        d = {}
        d["value"] = value
        converted = {}

        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                converted[TORCH_PREFIX + k] = v
            elif isinstance(v, np.ndarray):
                converted[NP_PREFIX + k] = torch.from_numpy(v)
            else:
                raise ValueError(
                    f"this only supports torch tensors and numpy arrays, found {type(v)}"
                )

        path = self.get_path(key)
        if not overwrite_if_exists and path.exists():
            raise FileExistsError(f"File already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(converted, path)

    def read(self, key):
        path = self.get_path(key)
        converted = safetensors.torch.load_file(path)
        data = {}
        for k, v in converted.items():
            if k.startswith(TORCH_PREFIX):
                data[k[len(TORCH_PREFIX) :]] = v
            elif k.startswith(NP_PREFIX):
                data[k[len(NP_PREFIX) :]] = v.numpy()
            else:
                raise RuntimeError("shouldn't happen")

        return data["value"]

    def exists(self, key):
        path = self.get_path(key)
        return path.exists()


# storer = Storer()


# key = "20260206/multitask_sparse_parity/parameter_scaling/v1/d_mlp=200/val_loss/numpy"
# # value = torch.tensor(0.123)
# value = np.array(0.123)


# # storer.write(key, value)

# print(storer.read(key))
