from pathlib import Path

import modal


def get_file_from_modal_volume_and_cache_locally(
    remote_path: str, local_root: str = "./modal_volume_cache", volume_name: str = "mtsp_results"
) -> Path:
    local_path = Path(local_root) / volume_name / remote_path

    if local_path.exists():
        # print(f"found locally. {local_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    vol = modal.Volume.from_name(volume_name)
    data = b"".join(vol.read_file(f"/{remote_path}"))
    local_path.write_bytes(data)
    # print(f"Downloaded {remote_path} -> {local_path}")

    return local_path


# get_file_from_modal_volume_and_cache_locally(
#     "metrics/20260209/train_mtsp_model_uniform_task_distribution/v10/1686079668477ac4/model/1000/losses_by_step_and_task.pt"
# )
