import json

from grounding.config import load_config


def test_project_root_is_relative_to_config_directory(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "config.json"
    path.write_text(json.dumps({"service": {"project_root": ".."}}), encoding="utf-8")
    config = load_config(path)
    assert config.service.project_root == str(tmp_path.resolve())
