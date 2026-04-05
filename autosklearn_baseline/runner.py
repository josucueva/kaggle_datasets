from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VALID_TASKS = ("classification", "regression")
VALID_TIERS = ("easy", "mid", "hard")
SKLEARN_METRICS_MODULE = "sklearn.metrics"


@dataclass(frozen=True)
class BaselineRunConfig:
    datasets_root: Path = Path("datasets")
    output_dir: Path = Path("results")
    target_config_path: Path = Path("autosklearn_baseline/dataset_targets.json")
    tasks: tuple[str, ...] = VALID_TASKS
    tiers: tuple[str, ...] = VALID_TIERS
    test_size: float = 0.2
    random_state: int = 42
    time_left_for_this_task: int = 180
    per_run_time_limit: int = 30


def run_baseline(config: BaselineRunConfig) -> tuple[pd.DataFrame, dict]:
    target_map = _load_target_map(config.target_config_path)

    discovered = _discover_datasets(
        datasets_root=config.datasets_root,
        tasks=config.tasks,
        tiers=config.tiers,
    )

    results = []
    for item in discovered:
        results.append(
            _run_one_dataset(item=item, config=config, target_map=target_map)
        )

    results_df = pd.DataFrame(results)
    summary = _build_summary(results_df)
    _save_outputs(results_df, summary, config.output_dir)
    return results_df, summary


def _discover_datasets(
    datasets_root: Path,
    tasks: Iterable[str],
    tiers: Iterable[str],
) -> list[dict]:
    found: list[dict] = []
    for task in tasks:
        for tier in tiers:
            tier_dir = datasets_root / task / tier
            if not tier_dir.exists():
                continue

            for dataset_dir in sorted([p for p in tier_dir.iterdir() if p.is_dir()]):
                csv_files = sorted(dataset_dir.rglob("*.csv"))
                found.append(
                    {
                        "task": task,
                        "tier": tier,
                        "dataset_name": dataset_dir.name,
                        "dataset_dir": dataset_dir,
                        "csv_files": csv_files,
                    }
                )
    return found


def _run_one_dataset(item: dict, config: BaselineRunConfig, target_map: dict) -> dict:
    train_test_split = _load_attr("sklearn.model_selection", "train_test_split")

    task = item["task"]
    tier = item["tier"]
    dataset_name = item["dataset_name"]
    dataset_dir = item["dataset_dir"]
    csv_files = item["csv_files"]

    base_result = {
        "task": task,
        "tier": tier,
        "dataset_name": dataset_name,
        "status": "failed",
        "error": "",
        "n_rows": np.nan,
        "n_cols": np.nan,
        "target_col": "",
        "target_source": "",
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "mae": np.nan,
        "mse": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "automl_models_tried": np.nan,
        "automl_ensemble_size": np.nan,
        "automl_best_model_weight": np.nan,
        "automl_best_model_name": "",
    }

    if len(csv_files) != 1:
        base_result["error"] = f"expected 1 csv, found {len(csv_files)}"
        return base_result

    try:
        df = pd.read_csv(csv_files[0])
        target_col, target_source, target_error = _resolve_target_column(
            df=df,
            task=task,
            dataset_name=dataset_name,
            target_map=target_map,
        )
        base_result["target_source"] = target_source

        if target_error:
            base_result["error"] = target_error
            return base_result

        if target_col is None:
            base_result["error"] = "target column not found"
            return base_result

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.empty:
            base_result["error"] = "feature matrix is empty"
            return base_result

        X = _encode_features_for_autosklearn(X)
        y = _prepare_target(y, task, df=df, target_col=target_col)

        if y.isna().any():
            valid_mask = ~y.isna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

        if len(X) < 10:
            base_result["error"] = "not enough rows after preprocessing"
            return base_result

        stratify = (
            y if task == "classification" and y.nunique(dropna=True) > 1 else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )

        model = _create_model(task=task, config=config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        base_result.update(_extract_model_metadata(model))

        base_result.update(
            {
                "status": "success",
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "target_col": target_col,
            }
        )

        if task == "classification":
            accuracy_score = _load_attr(SKLEARN_METRICS_MODULE, "accuracy_score")
            precision_score = _load_attr(SKLEARN_METRICS_MODULE, "precision_score")
            recall_score = _load_attr(SKLEARN_METRICS_MODULE, "recall_score")
            f1_score = _load_attr(SKLEARN_METRICS_MODULE, "f1_score")

            base_result.update(
                {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(
                        precision_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "f1": float(
                        f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    ),
                }
            )
        else:
            mean_absolute_error = _load_attr(
                SKLEARN_METRICS_MODULE, "mean_absolute_error"
            )
            mean_squared_error = _load_attr(
                SKLEARN_METRICS_MODULE, "mean_squared_error"
            )
            r2_score = _load_attr(SKLEARN_METRICS_MODULE, "r2_score")

            mse = float(mean_squared_error(y_test, y_pred))
            base_result.update(
                {
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "r2": float(r2_score(y_test, y_pred)),
                }
            )

        return base_result

    except Exception as exc:
        base_result["error"] = str(exc)
        return base_result


def _create_model(task: str, config: BaselineRunConfig):
    try:
        if task == "classification":
            autosklearn_classifier_cls = _load_attr(
                "autosklearn.classification", "AutoSklearnClassifier"
            )
            return autosklearn_classifier_cls(
                time_left_for_this_task=config.time_left_for_this_task,
                per_run_time_limit=config.per_run_time_limit,
                seed=config.random_state,
            )

        autosklearn_regressor_cls = _load_attr(
            "autosklearn.regression", "AutoSklearnRegressor"
        )
        return autosklearn_regressor_cls(
            time_left_for_this_task=config.time_left_for_this_task,
            per_run_time_limit=config.per_run_time_limit,
            seed=config.random_state,
        )
    except ImportError as exc:
        raise RuntimeError(
            "auto-sklearn is not installed. Install it with: pip install auto-sklearn"
        ) from exc


def _load_attr(module_name: str, attr_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"missing dependency module '{module_name}'. Install with: pip install -r requirements.txt"
        ) from exc
    return getattr(module, attr_name)


def _load_target_map(target_config_path: Path) -> dict:
    if not target_config_path.exists():
        return {}

    try:
        with target_config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}
    return raw


def _resolve_target_column(
    df: pd.DataFrame, task: str, dataset_name: str, target_map: dict
):
    configured_target = target_map.get(task, {}).get(dataset_name)
    if configured_target is not None:
        if configured_target in df.columns:
            return configured_target, "config", None
        return (
            None,
            "config",
            f"configured target '{configured_target}' not found in dataset columns",
        )

    inferred_target = _infer_target_column(df, task)
    return inferred_target, "heuristic", None


def _infer_target_column(df: pd.DataFrame, task: str) -> str | None:
    if df.empty:
        return None

    if task == "classification":
        hints = ("target", "label", "class", "outcome")
        hinted = _find_hinted_column(df.columns, hints)
        if hinted:
            return hinted

        candidates = [
            col
            for col in df.columns
            if 2 <= df[col].nunique(dropna=True) <= 50
            and df[col].nunique(dropna=True) < len(df)
        ]
        if candidates:
            return min(candidates, key=lambda c: df[c].nunique(dropna=True))
        return None

    hints = ("target", "y", "price", "salary", "score", "value")
    hinted = _find_hinted_column(df.select_dtypes(include=["number"]).columns, hints)
    if hinted:
        return hinted

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    return numeric_cols[-1] if numeric_cols else None


def _find_hinted_column(columns: Iterable[str], hints: tuple[str, ...]) -> str | None:
    for col in columns:
        normalized = str(col).lower().strip().replace(" ", "_")
        if normalized in hints:
            return str(col)
        for hint in hints:
            if normalized.startswith(f"{hint}_") or normalized.endswith(f"_{hint}"):
                return str(col)
    return None


def _encode_features_for_autosklearn(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        is_categorical = isinstance(out[col].dtype, pd.CategoricalDtype)
        if pd.api.types.is_object_dtype(out[col]) or is_categorical:
            out[col] = (
                out[col].astype("string").fillna("missing").astype("category").cat.codes
            )
    return out


def _prepare_target(
    y: pd.Series,
    task: str,
    df: pd.DataFrame | None = None,
    target_col: str | None = None,
) -> pd.Series:
    if task == "classification":
        return y.astype("string").fillna("missing").astype("category").cat.codes

    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.notna().sum() > 0:
        return y_numeric

    y_datetime = pd.to_datetime(y, errors="coerce")
    if y_datetime.notna().sum() == 0:
        return y_numeric

    if df is not None and "created_at" in df.columns:
        created_dt = pd.to_datetime(df["created_at"], errors="coerce")
        y_delta_minutes = (y_datetime - created_dt).dt.total_seconds() / 60.0
        return y_delta_minutes

    return y_numeric


def _build_summary(results_df: pd.DataFrame) -> dict:
    if results_df.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": 0,
            "success": 0,
            "failed": 0,
            "success_rate": 0.0,
            "classification": {},
            "regression": {},
            "datasets_latest": [],
        }

    success_df = results_df[results_df["status"] == "success"]
    total = int(len(results_df))
    success = int(len(success_df))
    failed = total - success

    cls_df = success_df[success_df["task"] == "classification"]
    reg_df = success_df[success_df["task"] == "regression"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": float(success / total) if total else 0.0,
        "classification": {
            "count": int(len(cls_df)),
            "accuracy_mean": _safe_mean(cls_df, "accuracy"),
            "precision_mean": _safe_mean(cls_df, "precision"),
            "recall_mean": _safe_mean(cls_df, "recall"),
            "f1_mean": _safe_mean(cls_df, "f1"),
            "models_tried_mean": _safe_mean(cls_df, "automl_models_tried"),
            "ensemble_size_mean": _safe_mean(cls_df, "automl_ensemble_size"),
        },
        "regression": {
            "count": int(len(reg_df)),
            "mae_mean": _safe_mean(reg_df, "mae"),
            "mse_mean": _safe_mean(reg_df, "mse"),
            "rmse_mean": _safe_mean(reg_df, "rmse"),
            "r2_mean": _safe_mean(reg_df, "r2"),
            "models_tried_mean": _safe_mean(reg_df, "automl_models_tried"),
            "ensemble_size_mean": _safe_mean(reg_df, "automl_ensemble_size"),
        },
        "datasets_latest": _build_datasets_latest(results_df),
    }


def _build_datasets_latest(results_df: pd.DataFrame) -> list[dict]:
    key_cols = ["task", "tier", "dataset_name"]
    if results_df.empty:
        return []

    counts = (
        results_df.groupby(key_cols, dropna=False).size().reset_index(name="runs_count")
    )

    latest_df = results_df.drop_duplicates(subset=key_cols, keep="last")
    latest_df = latest_df.merge(counts, on=key_cols, how="left")
    latest_df = latest_df.sort_values(key_cols)

    keep_cols = [
        "task",
        "tier",
        "dataset_name",
        "runs_count",
        "status",
        "error",
        "target_col",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mae",
        "mse",
        "rmse",
        "r2",
        "automl_models_tried",
        "automl_ensemble_size",
        "automl_best_model_weight",
        "automl_best_model_name",
    ]

    dataset_entries = []
    for _, row in latest_df.iterrows():
        entry = {col: _json_safe_value(row.get(col)) for col in keep_cols}
        dataset_entries.append(entry)

    return dataset_entries


def _json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _extract_model_metadata(model) -> dict:
    metadata = {
        "automl_models_tried": np.nan,
        "automl_ensemble_size": np.nan,
        "automl_best_model_weight": np.nan,
        "automl_best_model_name": "",
    }

    try:
        if hasattr(model, "cv_results_") and model.cv_results_:
            values = model.cv_results_.get("mean_test_score")
            if values is not None:
                metadata["automl_models_tried"] = int(len(values))
    except Exception:
        pass

    try:
        if hasattr(model, "get_models_with_weights"):
            models_with_weights = model.get_models_with_weights()
            if models_with_weights:
                metadata["automl_ensemble_size"] = int(len(models_with_weights))
                best_weight, best_model = max(models_with_weights, key=lambda x: x[0])
                metadata["automl_best_model_weight"] = float(best_weight)
                metadata["automl_best_model_name"] = type(best_model).__name__
    except Exception:
        pass

    return metadata


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    val = df[col].dropna().mean()
    return float(val) if pd.notna(val) else None


def _save_outputs(results_df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "autosklearn_baseline_results.csv"
    summary_path = output_dir / "autosklearn_baseline_summary.json"

    combined_df = _append_results(results_path, results_df)
    cumulative_summary = _build_summary(combined_df)
    cumulative_summary["last_run"] = summary

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(cumulative_summary, f, indent=2)


def _append_results(results_path: Path, new_results_df: pd.DataFrame) -> pd.DataFrame:
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
    else:
        existing_df = pd.DataFrame()

    combined_df = pd.concat(
        [existing_df, new_results_df], ignore_index=True, sort=False
    )
    combined_df.to_csv(results_path, index=False)
    return combined_df
