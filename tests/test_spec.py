from __future__ import annotations as _annotations

from pathlib import Path

import pytest

from autobench.errors import SpecValidationError
from autobench.spec import load_benchmark_spec


def test_load_benchmark_spec_accepts_minimal_spec(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text("benchmark:\n  id: minimal\n", encoding="utf-8")

    spec = load_benchmark_spec(path)

    assert spec.benchmark.id == "minimal"


def test_load_benchmark_spec_rejects_missing_benchmark_id(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text("benchmark:\n  description: missing id\n", encoding="utf-8")

    with pytest.raises(SpecValidationError):
        load_benchmark_spec(path)
