from __future__ import annotations as _annotations

import sys
from pathlib import Path

import pytest


def _prepend_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


ROOT = Path(__file__).resolve().parents[1]

_prepend_path(ROOT)
_prepend_path(ROOT / "src")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "asyncio_mode",
        "Compatibility shim for pytest-asyncio config when plugin autoload is disabled.",
        default="auto",
    )
