import tomllib
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DependencySource(TypedDict, total=False):
    directory: str
    editable: str | bool
    git: str
    path: str
    rev: str


class UVConfiguration(TypedDict):
    sources: dict[str, DependencySource]


class ToolConfiguration(TypedDict):
    uv: UVConfiguration


class ProjectConfiguration(TypedDict):
    tool: ToolConfiguration


class LockedPackage(TypedDict):
    name: str
    source: DependencySource


class LockConfiguration(TypedDict):
    package: list[LockedPackage]


def test_dependency_sources_are_portable_from_a_standalone_checkout() -> None:
    project = cast(
        ProjectConfiguration,
        tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")),
    )
    lock = cast(
        LockConfiguration,
        tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")),
    )
    sources = [*project["tool"]["uv"]["sources"].values()]
    sources.extend(package["source"] for package in lock["package"])

    for source in sources:
        for key in ("directory", "editable", "path"):
            value = source.get(key)
            if not isinstance(value, str):
                continue
            source_path = PurePosixPath(value)
            assert not source_path.is_absolute(), f"dependency source must be relative: {value}"
            assert ".." not in source_path.parts, (
                f"dependency source must stay inside the repository: {value}"
            )
