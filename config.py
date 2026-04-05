from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class AutoSettings:
    default_tasks: Tuple[str, ...] = ("classification", "regression")
    valid_tasks: Tuple[str, ...] = ("classification", "regression")
    default_count: int = 10
    difficulty_levels: Tuple[str, ...] = ("easy", "mid", "hard")
    search_suffixes: Tuple[str, ...] = ("", " dataset", " data")
    sort_by: str = "hottest"
    file_type: str = "csv"
    min_size: int = 1_000_000  # 1MB
    max_size: int = 50_000_000  # 50MB
    search_pages: int = 5
    relax_on_missing: bool = True
    relaxed_search_suffixes: Tuple[str, ...] = (
        "",
        " dataset",
        " data",
        " machine learning",
        " prediction",
        " tabular",
    )
    relaxed_search_pages: int = 8
    allow_multi_csv_on_relaxed: bool = True


@dataclass(frozen=True)
class UrlSettings:
    default_url_file: str = "dataset_urls.txt"
    kaggle_protocol_prefix: str = "kaggle://"


@dataclass(frozen=True)
class AnalyzerSettings:
    easy_max_columns: int = 15
    easy_max_class_majority: float = 0.60
    easy_required_missing_ratio: float = 0.0
    easy_required_invalid_ratio: float = 0.0
    easy_require_numeric_only: bool = True

    hard_min_points: int = 6

    moderate_missing_ratio: float = 0.05
    high_missing_ratio: float = 0.20
    moderate_invalid_ratio: float = 0.01
    high_invalid_ratio: float = 0.05
    low_numeric_ratio: float = 0.50
    medium_numeric_ratio: float = 0.80
    medium_columns: int = 40
    high_columns: int = 100
    medium_feature_sample_ratio: float = 0.10
    high_feature_sample_ratio: float = 0.20
    medium_categorical_cardinality: int = 80
    high_categorical_cardinality: int = 200
    moderate_imbalance: float = 0.65
    high_imbalance: float = 0.75
    severe_imbalance: float = 0.90
    medium_multiclass: int = 10
    high_multiclass: int = 20


@dataclass(frozen=True)
class AppSettings:
    auto: AutoSettings = AutoSettings()
    url: UrlSettings = UrlSettings()
    analyzer: AnalyzerSettings = AnalyzerSettings()


settings = AppSettings()


def validate_tasks(tasks: Iterable[str]) -> list[str]:
    invalid = sorted(set(tasks) - set(settings.auto.valid_tasks))
    if invalid:
        raise ValueError(
            f"invalid task(s): {', '.join(invalid)}. "
            f"valid tasks: {', '.join(settings.auto.valid_tasks)}"
        )
    return list(tasks)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be a positive integer")
    return parsed
