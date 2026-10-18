"""Tests for configuration-controlled pipeline paths."""
from __future__ import annotations

from pathlib import Path

from tsnd.cli import _dirs
from tsnd.models.alignment import resolve_artifact_dir as resolve_alignment_dir
from tsnd.models.embeddings import resolve_artifact_dir as resolve_embedding_dir
from tsnd.models.specter_embed import resolve_artifact_dir as resolve_specter_dir


def test_dirs_honors_explicit_artifact_root(tmp_path: Path) -> None:
    """Use ``artifacts.out_dir`` instead of deriving a root from ``run_tag``."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "run.yaml"
    config_path.write_text("run_tag: ignored\n", encoding="utf-8")
    config = {
        "run_tag": "ignored",
        "data": {"out_dir": "data_locked"},
        "artifacts": {"out_dir": "artifacts_locked"},
    }

    data_dir, artifact_dir = _dirs(config, str(config_path))

    assert data_dir == tmp_path / "data_locked"
    assert artifact_dir == tmp_path / "artifacts_locked"


def test_stage_modules_share_explicit_artifact_root(tmp_path: Path) -> None:
    """Resolve the same configured artifact root in every standalone stage."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "run.yaml"
    config = {
        "run_tag": "ignored",
        "artifacts": {"out_dir": "artifacts_locked"},
    }
    expected = tmp_path / "artifacts_locked"

    assert resolve_embedding_dir(config, config_path) == expected
    assert resolve_alignment_dir(config, config_path) == expected
    assert resolve_specter_dir(config, config_path) == expected
