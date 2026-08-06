#!/usr/bin/env python3
"""Inspect Autobench and optional native-instrumentation dependencies."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypedDict


class PackageStatus(TypedDict):
    distribution: str
    integration: str
    installed: bool
    version: str | None


INTEGRATIONS: tuple[tuple[str, str], ...] = (
    ("autobench", "core"),
    ("pydantic-ai-slim", "pydantic_ai"),
    ("openai", "openai"),
    ("openai-agents", "openai_agents"),
    ("httpx", "httpx"),
)


def package_status(distribution: str, integration: str) -> PackageStatus:
    """Return one installed-distribution status without importing the package."""
    try:
        installed_version = version(distribution)
    except PackageNotFoundError:
        installed_version = None
    return PackageStatus(
        distribution=distribution,
        integration=integration,
        installed=installed_version is not None,
        version=installed_version,
    )


def find_specs(root: Path) -> list[str]:
    """Find likely benchmark specs under a project root."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.yaml")
        if path.name in {"autobench.yaml", "benchmark.yaml"}
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def main() -> None:
    """Render environment status as stable JSON for humans or agents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    payload = {
        "root": str(root),
        "packages": [package_status(name, integration) for name, integration in INTEGRATIONS],
        "benchmark_specs": find_specs(root),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
