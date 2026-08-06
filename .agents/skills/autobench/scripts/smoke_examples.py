#!/usr/bin/env python3
"""Run every offline YAML example bundled with the Autobench skill."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

OFFLINE_EXAMPLES = ("abp-manual", "minimal", "performance")


def main() -> None:
    """Validate, execute, replay, and report each offline example."""
    examples = Path(__file__).resolve().parents[1] / "examples"
    with tempfile.TemporaryDirectory(prefix="autobench-skill-") as temporary:
        root = Path(temporary)
        for name in OFFLINE_EXAMPLES:
            spec = examples / name / "autobench.yaml"
            record = root / name
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("validate_workflow.py")),
                    str(spec),
                    "--record",
                    str(record),
                ],
                check=True,
            )
    print(f"Validated {len(OFFLINE_EXAMPLES)} offline Autobench skill examples.")


if __name__ == "__main__":
    main()
