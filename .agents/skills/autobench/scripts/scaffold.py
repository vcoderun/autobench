#!/usr/bin/env python3
"""Copy one maintained Autobench skill example into a new directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXAMPLES = ("abp-manual", "minimal", "performance", "pydantic-ai", "pydantic-gepa")


def scaffold(example: str, destination: Path) -> None:
    """Copy a complete example without overwriting an existing destination."""
    source = Path(__file__).resolve().parents[1] / "examples" / example
    if not source.is_dir():
        raise SystemExit(f"Skill example does not exist: {source}")
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def main() -> None:
    """Parse CLI arguments and scaffold the selected example."""
    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=EXAMPLES)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    scaffold(args.example, destination)
    print(destination)
    if (destination / "autobench.yaml").is_file():
        print(f"autobench validate {destination / 'autobench.yaml'}")


if __name__ == "__main__":
    main()
