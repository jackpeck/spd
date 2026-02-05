from pathlib import Path

import wandb


def load_wandb_model_artifact(
    run_path: str, filename: str = "model.pt", cache_dir: Path = Path("/tmp") / "wandb_local"
):
    """Download a wandb model artifact, caching locally to skip API calls if cached. Save ~2s."""
    artifact_dir = cache_dir / run_path
    filepath = artifact_dir / filename
    if not filepath.exists():
        artifact_dir.mkdir(parents=True, exist_ok=True)
        api = wandb.Api()
        run = api.run(run_path)
        artifact = [a for a in run.logged_artifacts() if a.type == "model"][-1]
        artifact.download(root=artifact_dir)
    return filepath
