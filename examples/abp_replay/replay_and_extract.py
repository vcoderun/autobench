from __future__ import annotations as _annotations

import argparse
from pathlib import Path

from autobench import CompositeExtractor, SignalExtractor, SpanExtractor, UsageExtractor
from autobench.records.replay import load_experiment_record, load_run_record, replay_extraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    record = load_experiment_record(args.run_dir)
    extractor = CompositeExtractor(SignalExtractor(), SpanExtractor(), UsageExtractor())
    for relative_path in record.run_paths:
        run = load_run_record(args.run_dir / relative_path, root_dir=args.run_dir)
        derived = replay_extraction(run, extractor)
        print(derived.run_id, len(derived.observations), derived.parent_run_id)


if __name__ == "__main__":
    main()
