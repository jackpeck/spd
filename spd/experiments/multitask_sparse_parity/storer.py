import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from modal_utils import file_exists_on_modal_volume, get_file_from_modal_volume_and_cache_locally

TORCH_PREFIX = "torch:"
NP_PREFIX = "np:"
DICT_PREFIX = "dict:"
dict_sentinel = "__dict__sentinel"
value_suffix = "value"


local_modal_cache_root = Path("./modal_volume_cache")


class Storer:
    def __init__(self, root_path: Path | str = Path("./metrics/")):
        self.root_path = root_path
        if isinstance(self.root_path, str):
            self.root_path = Path(self.root_path)

        self.under_modal_volume = self.root_path.parts[:2] == ("/", "modal_volume")
        self.on_modal = os.environ.get("MODAL_ENVIRONMENT") is not None

        if self.under_modal_volume:
            modal_volume_name = self.root_path.parts[2]
            modal_volume_path_parts = self.root_path.parts[3:]
            if self.on_modal:
                self.root_path = Path("/", modal_volume_name, *modal_volume_path_parts)
            else:
                self.root_path = Path(
                    "./modal_volume_cache", modal_volume_name, *modal_volume_path_parts
                )

    def get_modal_path_from_local_path(self, path):
        assert path.is_relative_to(
            local_modal_cache_root
        )  # assert path is under local_model_cache_root
        parts = path.parts[len(local_modal_cache_root.parts) :]
        modal_volume_name = parts[0]
        modal_volume_path_parts = parts[1:]
        # modal_path = Path("/", modal_volume_name, *modal_volume_path_parts)
        # return modal_path
        return "/".join(modal_volume_path_parts)

    def __repr__(self):
        return f"{self.__class__.__name__}(root_path={self.root_path!r})"

    def add_prefix(self, prefix):
        self.root_path = self.root_path / prefix

    def add_datestamp_prefix(self):
        datestamp = datetime.now(timezone(timedelta(hours=-8))).strftime("%Y%m%d")
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

        self.pull_locally_if_modal_volume(key)
        path = self.get_path(key)
        if not overwrite_if_exists and path.exists():
            raise FileExistsError(f"File already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(data, path)

    def read(self, key):
        # print(key)
        # print(self.exists(key))

        self.pull_locally_if_modal_volume(key)

        path = self.get_path(key)

        # print(path)
        # print("opiuoi", path.exists())

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

    def pull_locally_if_modal_volume(self, key):
        if self.under_modal_volume and not self.on_modal:
            path = self.get_path(key)
            if not path.exists():
                modal_path = self.get_modal_path_from_local_path(path)
                if file_exists_on_modal_volume(modal_path, "mtsp_results"):
                    get_file_from_modal_volume_and_cache_locally(
                        modal_path, "./modal_volume_cache", "mtsp_results"
                    )

    def exists_on_modal(self, key):
        path = self.get_path(key)
        modal_path = self.get_modal_path_from_local_path(path)
        exists_remotely = file_exists_on_modal_volume(modal_path, "mtsp_results")
        return exists_remotely

    def exists(self, key):
        path = self.get_path(key)

        if path.exists():
            return True

        should_check_modal = self.under_modal_volume and not self.on_modal
        return should_check_modal and self.exists_on_modal(key)


# storer = Storer()


# key = "20260206/multitask_sparse_parity/parameter_scaling/v1/d_mlp=200/val_loss/numpy"
# # value = torch.tensor(0.123)
# value = np.array(0.123)


# # storer.write(key, value)

# print(storer.read(key))
