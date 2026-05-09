from __future__ import annotations as _annotations

from pathlib import Path

import conftest


def test_prepend_path_adds_missing_path(tmp_path: Path) -> None:
    helper_path = tmp_path / "helper-src"
    helper_path.mkdir()

    conftest._prepend_path(helper_path)

    assert str(helper_path.resolve()) in conftest.sys.path


def test_prepend_path_leaves_existing_path_in_place(tmp_path: Path) -> None:
    helper_path = tmp_path / "existing-src"
    helper_path.mkdir()
    resolved = str(helper_path.resolve())
    conftest.sys.path.insert(0, resolved)

    conftest._prepend_path(helper_path)

    assert conftest.sys.path.count(resolved) == 1
