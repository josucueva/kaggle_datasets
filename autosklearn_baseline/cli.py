import argparse
import importlib
import sys
from pathlib import Path

from autosklearn_baseline.runner import (
    BaselineRunConfig,
    VALID_TASKS,
    VALID_TIERS,
    run_baseline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run default auto-sklearn baseline over local tiered datasets."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=VALID_TASKS,
        default=list(VALID_TASKS),
        help="Task(s) to run.",
    )
    parser.add_argument(
        "--tier",
        nargs="+",
        choices=VALID_TIERS,
        default=list(VALID_TIERS),
        help="Tier(s) to run.",
    )
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        help="Root folder for tiered datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Folder where CSV/JSON reports are written.",
    )
    parser.add_argument(
        "--target-config",
        default="autosklearn_baseline/dataset_targets.json",
        help="Path to JSON file mapping dataset_name to target column per task.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--time-left",
        type=int,
        default=180,
        help="Total time budget per dataset in seconds.",
    )
    parser.add_argument(
        "--per-run-time-limit",
        type=int,
        default=30,
        help="Time limit per model run in seconds.",
    )

    args = parser.parse_args()

    _check_runtime_dependencies()

    config = BaselineRunConfig(
        datasets_root=Path(args.datasets_root),
        output_dir=Path(args.output_dir),
        target_config_path=Path(args.target_config),
        tasks=tuple(args.task),
        tiers=tuple(args.tier),
        test_size=args.test_size,
        random_state=args.seed,
        time_left_for_this_task=args.time_left,
        per_run_time_limit=args.per_run_time_limit,
    )

    results_df, summary = run_baseline(config)

    print("\nautosklearn baseline run completed")
    print(f"total datasets: {summary['total']}")
    print(f"successful: {summary['success']}")
    print(f"failed: {summary['failed']}")
    print(f"success rate: {summary['success_rate']:.2%}")

    if not results_df.empty:
        failed = results_df[results_df["status"] == "failed"]
        if not failed.empty:
            print("\nfailed datasets:")
            for _, row in failed.iterrows():
                print(
                    f"- {row['task']}/{row['tier']}/{row['dataset_name']}: {row['error']}"
                )


def _check_runtime_dependencies() -> None:
    if sys.version_info >= (3, 10):
        print("auto-sklearn baseline requires Python 3.9 in this project setup.")
        print("current version:", sys.version.split()[0])
        print("\ncreate and activate a compatible env:")
        print("uv venv -p 3.9 .venv-autosklearn")
        print("source .venv-autosklearn/bin/activate")
        print("uv pip install swig")
        print("uv pip install -r requirements.txt")
        raise SystemExit(2)

    required_modules = [
        "sklearn",
        "autosklearn.classification",
        "autosklearn.regression",
    ]

    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        print("missing dependencies for baseline execution:")
        for module_name in missing:
            print(f"- {module_name}")
        print("\ninstall dependencies with:")
        print("pip install -r requirements.txt")
        print("\nauto-sklearn may require Python <= 3.11 on many systems.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
