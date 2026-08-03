from __future__ import annotations as _annotations

import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import autobench
from autobench import Benchmark, record_experiment, replay_experiment

_OPTIONAL_MODULES = ("agents", "httpx", "openai", "pydantic_ai")


def main() -> None:
    imported_optional = tuple(name for name in _OPTIONAL_MODULES if name in sys.modules)
    if imported_optional:
        raise RuntimeError(
            f"core wheel imported optional SDK modules: {', '.join(imported_optional)}"
        )
    installed_version = version("autobench")
    if autobench.__version__ != installed_version:
        raise RuntimeError(
            f"package version mismatch: module={autobench.__version__}, metadata={installed_version}"
        )

    result = Benchmark("wheel-smoke").run(experiment_id="exp_wheel_smoke")
    with TemporaryDirectory(prefix="autobench-wheel-smoke-") as directory:
        record_dir = Path(directory) / "record"
        record_experiment(result, record_dir)
        replayed = replay_experiment(record_dir)
    if replayed.experiment_id != result.experiment_id or replayed.runs:
        raise RuntimeError("core wheel record/replay smoke did not preserve the experiment")

    print("core import, optional dependency isolation, record, and replay passed")


if __name__ == "__main__":
    main()
