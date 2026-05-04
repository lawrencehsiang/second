from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = [
    "gsm8k",
    "math",
    "multiarith",
    "addsub",
    "asdiv",
    "singleeq",
    "svamp",
]


def run_one_dataset(
    dataset: str,
    limit: int,
    max_round: int,
    output_root: Path,
    stop_on_error: bool,
) -> bool:
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--dataset",
        dataset,
        "--limit",
        str(limit),
        "--max-round",
        str(max_round),
        "--output-dir",
        str(output_dir),
    ]

    print("=" * 80)
    print(f"Running dataset: {dataset}")
    print(f"Output dir: {output_dir}")
    print(f"Log file: {log_path}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"Running dataset: {dataset}\n")
        log_file.write("Command: " + " ".join(cmd) + "\n")
        log_file.write("=" * 80 + "\n")

        process = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if process.returncode == 0:
        print(f"[OK] Finished dataset: {dataset}")
        return True

    print(f"[FAILED] Dataset failed: {dataset}. See log: {log_path}")

    if stop_on_error:
        raise RuntimeError(f"Dataset failed: {dataset}")

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comparison experiments for multiple datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of samples per dataset.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=5,
        help="Maximum debate rounds.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one dataset fails.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    succeeded = []
    failed = []

    for dataset in args.datasets:
        ok = run_one_dataset(
            dataset=dataset,
            limit=args.limit,
            max_round=args.max_round,
            output_root=output_root,
            stop_on_error=args.stop_on_error,
        )

        if ok:
            succeeded.append(dataset)
        else:
            failed.append(dataset)

    print("\n" + "=" * 80)
    print("All requested datasets have finished.")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Output root: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()