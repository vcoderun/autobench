from __future__ import annotations as _annotations

import sys
import tarfile
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

    source_distributions = tuple(Path("dist").glob("*.tar.gz"))
    if len(source_distributions) != 1:
        raise RuntimeError("wheel smoke requires exactly one source distribution")
    forbidden_prefixes = (
        ".agents/",
        ".autobench/",
        ".cache/",
        ".codex/",
        ".env",
        ".pytest_cache/",
        ".ruff_cache/",
        ".venv/",
        "AGENTS.md",
        "COVERAGE",
        "docs/plans/",
        "mocks/",
        "plans/",
        "references/",
        "runs/",
    )
    with tarfile.open(source_distributions[0], mode="r:gz") as source_archive:
        leaked_paths = []
        for archive_path in source_archive.getnames():
            path_parts = archive_path.split("/", maxsplit=1)
            relative_path = path_parts[1] if len(path_parts) == 2 else path_parts[0]
            if any(
                relative_path == prefix.rstrip("/") or relative_path.startswith(prefix)
                for prefix in forbidden_prefixes
            ):
                leaked_paths.append(relative_path)
    if leaked_paths:
        raise RuntimeError(
            "source distribution contains private or generated paths: "
            + ", ".join(sorted(leaked_paths))
        )

    result = Benchmark("wheel-smoke").run(experiment_id="exp_wheel_smoke")
    with TemporaryDirectory(prefix="autobench-wheel-smoke-") as directory:
        record_dir = Path(directory) / "record"
        record_experiment(result, record_dir)
        replayed = replay_experiment(record_dir)
    if replayed.experiment_id != result.experiment_id or replayed.runs:
        raise RuntimeError("core wheel record/replay smoke did not preserve the experiment")

    print("distribution hygiene, core import, dependency isolation, record, and replay passed")


if __name__ == "__main__":
    main()
