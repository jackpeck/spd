from datetime import datetime
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

TORCH_PREFIX = "torch:"
NP_PREFIX = "np:"
DICT_PREFIX = "dict:"
dict_sentinel = "__dict__sentinel"
value_suffix = "value"


class Storer:
    def __init__(self, root_path=Path("./metrics/")):
        self.root_path = root_path

    def __repr__(self):
        return f"{self.__class__.__name__}(root_path={self.root_path!r})"

    def add_prefix(self, prefix):
        self.root_path = self.root_path / prefix

    def add_datestamp_prefix(self):
        datestamp = datetime.now().strftime("%Y%m%d")
        self.add_prefix(datestamp)

    def get_path(self, key):
        return self.root_path / (key + ".pt")

    def write(self, key, value, overwrite_if_exists=False):
        data = None
        if isinstance(value, dict):
            data = value
            data[dict_sentinel] = torch.tensor(0)
        else:
            data = {}
            # d["value"] = value
            if isinstance(value, torch.Tensor):
                data[TORCH_PREFIX + value_suffix] = value
            elif isinstance(value, np.ndarray):
                data[NP_PREFIX + value_suffix] = torch.from_numpy(value)
            else:
                raise ValueError(
                    f"this only supports torch tensors and numpy arrays and dicts, found {type(value)}"
                )

        path = self.get_path(key)
        if not overwrite_if_exists and path.exists():
            raise FileExistsError(f"File already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(data, path)

    def read(self, key):
        path = self.get_path(key)
        converted = safetensors.torch.load_file(path)

        if dict_sentinel in converted:
            del converted[dict_sentinel]
            return converted
        else:
            if (TORCH_PREFIX + value_suffix) in converted:
                return converted[(TORCH_PREFIX + value_suffix)]
            elif (NP_PREFIX + value_suffix) in converted:
                return converted[(NP_PREFIX + value_suffix)].numpy()
            else:
                raise RuntimeError("shouldn't happen")

    def exists(self, key):
        path = self.get_path(key)
        return path.exists()


# storer = Storer()


# key = "20260206/multitask_sparse_parity/parameter_scaling/v1/d_mlp=200/val_loss/numpy"
# # value = torch.tensor(0.123)
# value = np.array(0.123)


# # storer.write(key, value)

# print(storer.read(key))
