"""Model provisioning commands for local and hosted deployments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

from .config import load_config, project_path
from .ultralytics_env import configure_ultralytics_directory


DEFAULT_OWLVIT_REPOSITORY = "google/owlvit-base-patch32"
DEFAULT_OWLVIT_REVISION = "cbc355fb364588351c5d51c7f74465e8e7ec6f72"
DEFAULT_YOLO_MODEL = "yolo11n.pt"


def provision_owlvit(
    destination: str | Path = "models/owlvit/owlvit-base-patch32",
    *,
    repository: str = DEFAULT_OWLVIT_REPOSITORY,
    revision: str = DEFAULT_OWLVIT_REVISION,
) -> Path:
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=str(target),
        ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
    )
    return target


def provision_yolo(
    destination: str | Path = "models/yolo/detector.pt",
    *,
    model_name: str = DEFAULT_YOLO_MODEL,
) -> Path:
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    configure_ultralytics_directory()
    from ultralytics import YOLO

    model = YOLO(model_name)
    source = Path(getattr(model, "ckpt_path", model_name)).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"Ultralytics did not provide the requested model: {model_name}")
    if source != target:
        shutil.copy2(source, target)
        if source.name == model_name and source.parent == Path.cwd().resolve():
            source.unlink(missing_ok=True)
    return target


def provision_enabled_models(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    created: dict[str, Path] = {}

    yolo_settings = config.backends.yolo
    yolo_path = project_path(config, yolo_settings.weights_path)
    if yolo_settings.enabled and yolo_path is not None and not yolo_path.exists():
        created["yolo"] = provision_yolo(
            yolo_path,
            model_name=os.environ.get("GROUNDING_YOLO_MODEL", DEFAULT_YOLO_MODEL),
        )

    owlvit_settings = config.backends.owlvit
    owlvit_path = project_path(config, owlvit_settings.model_path)
    owlvit_weight = owlvit_path / "model.safetensors" if owlvit_path else None
    if owlvit_settings.enabled and owlvit_weight is not None and not owlvit_weight.exists():
        created["owlvit"] = provision_owlvit(
            owlvit_path,
            repository=os.environ.get(
                "GROUNDING_OWLVIT_REPOSITORY",
                DEFAULT_OWLVIT_REPOSITORY,
            ),
            revision=os.environ.get(
                "GROUNDING_OWLVIT_REVISION",
                DEFAULT_OWLVIT_REVISION,
            ),
        )
    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Provision local grounding models")
    parser.add_argument(
        "--config",
        default="configs/grounding_service.json",
        help="Configuration whose enabled model paths should be provisioned",
    )
    parser.add_argument("--owlvit-only", action="store_true")
    parser.add_argument("--yolo-only", action="store_true")
    args = parser.parse_args(argv)

    if args.owlvit_only and args.yolo_only:
        parser.error("choose at most one model-specific option")

    provisioned: dict[str, Any]
    if args.owlvit_only:
        provisioned = {"owlvit": provision_owlvit()}
    elif args.yolo_only:
        provisioned = {"yolo": provision_yolo()}
    else:
        provisioned = provision_enabled_models(args.config)

    if not provisioned:
        print("All enabled model files are already present.")
        return
    for name, path in provisioned.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
