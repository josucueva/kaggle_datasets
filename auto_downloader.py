import os
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
from dataset_analyzer import analizar_dataframe, determinar_nivel, print_metrics
from utils import (
    has_single_csv,
    get_csv_path,
    read_dataset_sample,
    copy_dataset,
    ensure_dir,
)


def run_auto_collection(tasks=None, target_per_level=1):
    """
    auto download and organize datasets by difficulty

    args:
        tasks: list of tasks to collect (default: ['classification', 'regression'])
        target_per_level: number of datasets per difficulty level
    """
    if tasks is None:
        tasks = ["classification", "regression"]

    api = KaggleApi()
    api.authenticate()

    print("kaggle dataset collector for automl")
    print("filtering: single csv files only\n")

    for tarea in tasks:
        print(f"\n{'='*50}")
        print(f"task: {tarea}")
        print(f"{'='*50}")

        datasets = api.dataset_list(
            search=tarea,
            sort_by="hottest",
            file_type="csv",
            min_size="10000000",
            max_size="100000000",
        )

        if not datasets:
            print("no datasets found")
            continue

        datasets_por_nivel = {"easy": 0, "mid": 0, "hard": 0}

        print(f"\nsearching {target_per_level} dataset(s) per difficulty level...\n")

        for d in datasets:
            if all(count >= target_per_level for count in datasets_por_nivel.values()):
                print(f"\ncompleted: {target_per_level} datasets per level")
                break

            if d is None:
                continue

            print(f"\nanalyzing: {d.ref}")
            try:
                temp_path = kagglehub.dataset_download(d.ref)

                is_single, _ = has_single_csv(temp_path)
                if not is_single:
                    csv_count = len(_)
                    print(f"   skipping: {csv_count} csv files found, need exactly 1")
                    continue

                csv_path = get_csv_path(temp_path)
                df = read_dataset_sample(csv_path)

                metricas = analizar_dataframe(df, tarea)
                nivel, razones = determinar_nivel(metricas, tarea)

                if datasets_por_nivel.get(nivel, 0) >= target_per_level:
                    print(f"   skipping: already have enough '{nivel}' datasets")
                    continue

                dataset_dir = os.path.join(os.getcwd(), "datasets", tarea, nivel)
                ensure_dir(dataset_dir)

                dataset_name = d.ref.replace("/", "_")
                final_path = os.path.join(dataset_dir, dataset_name)

                copy_dataset(temp_path, final_path)

                datasets_por_nivel[nivel] = datasets_por_nivel.get(nivel, 0) + 1

                print_metrics(metricas, nivel, razones)
                print(f"   saved to: datasets/{tarea}/{nivel}/")
                print(f"   progress: {datasets_por_nivel}")

            except Exception as e:
                print(f"   error: {e}")

    _print_summary()


def _print_summary():
    """print final summary"""
    print(f"\n{'='*50}")
    print("summary")
    print(f"{'='*50}")
    print("\nprocess completed")
    print("\ndatasets organized in:")
    print("  datasets/classification/{easy,mid,hard}/")
    print("  datasets/regression/{easy,mid,hard}/")
    print("\nready for automl testing\n")
