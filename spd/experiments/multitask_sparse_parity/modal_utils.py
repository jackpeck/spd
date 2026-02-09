from pathlib import Path, PurePosixPath

import modal


def file_exists_on_modal_volume(remote_path, volume_name):
    volume = modal.Volume.from_name(volume_name)
    parent = str(PurePosixPath(remote_path).parent)
    try:
        return any(e.path == remote_path for e in volume.listdir(parent))
    except modal.exception.NotFoundError:
        return False


def get_file_from_modal_volume_and_cache_locally(
    remote_path: str, local_root: str = "./modal_volume_cache", volume_name: str = "mtsp_results"
) -> Path:
    local_path = Path(local_root) / volume_name / remote_path

    if local_path.exists():
        # print(f"found locally. {local_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    volume = modal.Volume.from_name(volume_name)
    data = b"".join(volume.read_file(f"/{remote_path}"))
    local_path.write_bytes(data)
    # print(f"Downloaded {remote_path} -> {local_path}")

    return local_path


# get_file_from_modal_volume_and_cache_locally(
#     "metrics/20260209/train_mtsp_model_uniform_task_distribution/v10/1686079668477ac4/model/1000/losses_by_step_and_task.pt"
# )
