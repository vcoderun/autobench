#!/usr/bin/env python3
"""Validate a benchmark and optionally execute its record/replay/report workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(arguments: list[str]) -> None:
    """Run one Autobench CLI command with the active Python environment."""
    subprocess.run([sys.executable, "-m", "autobench", *arguments], check=True)


def validate_workflow(spec: Path, record: Path | None) -> None:
    """Validate a spec and run offline record consumers when a record path is provided."""
    run_command(["validate", str(spec)])
    if record is None:
        return
    run_command(["run", str(spec), "--record", str(record)])
    run_command(["replay", str(record)])
    run_command(["report", str(record)])


def main() -> None:
    """Parse CLI arguments and execute the selected validation depth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    spec = args.spec.resolve()
    if not spec.is_file():
        raise SystemExit(f"Benchmark spec does not exist: {spec}")
    record = args.record.resolve() if args.record is not None else None
    validate_workflow(spec, record)


if __name__ == "__main__":
    main()
