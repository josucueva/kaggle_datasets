import os
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
from config import settings, validate_tasks
from dataset_analyzer import analizar_dataframe, determinar_nivel, print_metrics
from utils import (
    has_single_csv,
    get_csv_path,
    read_dataset_sample,
    copy_dataset,
    ensure_dir,
)


def run_auto_collection(tasks=None, target_per_level=5, tiers=None):
    """
    auto download and organize datasets by difficulty

    args:
        tasks: list of tasks to collect (default: ['classification', 'regression'])
        target_per_level: number of datasets per difficulty level
        tiers: difficulty levels to collect (default: easy, mid, hard)
    """
    if tasks is None:
        tasks = list(settings.auto.default_tasks)
    if tiers is None:
        tiers = list(settings.auto.difficulty_levels)

    tasks = validate_tasks(tasks)
    selected_levels = [
        level for level in settings.auto.difficulty_levels if level in tiers
    ]
    if not selected_levels:
        print("no valid tiers selected")
        return

    api = KaggleApi()
    api.authenticate()
    max_size = int(settings.auto.max_size)
    print(f"max dataset size cap: {max_size} bytes")

    print("kaggle dataset collector for automl")
    print("filtering: single csv files only\n")

    for tarea in tasks:
        print(f"\n{'='*50}")
        print(f"task: {tarea}")
        print(f"{'='*50}")

        datasets = _get_candidate_datasets(api, tarea, relaxed=False)

        if not datasets:
            print("no datasets found")
            continue

        # initialize counters from existing dataset folders so we only fill pending tiers
        datasets_por_nivel = _count_existing_by_level(tarea, selected_levels)

        print("existing datasets by level:")
        print(f"   {datasets_por_nivel}")

        print(f"\nsearching {target_per_level} dataset(s) per difficulty level...\n")

        processed_refs = set()
        _process_candidates(
            api=api,
            datasets=datasets,
            task=tarea,
            target_per_level=target_per_level,
            datasets_por_nivel=datasets_por_nivel,
            processed_refs=processed_refs,
            allow_multi_csv=False,
            selected_levels=selected_levels,
            max_size=max_size,
            size_cache={},
        )

        missing = {
            nivel: target_per_level - count
            for nivel, count in datasets_por_nivel.items()
            if count < target_per_level
        }

        if missing and settings.auto.relax_on_missing:
            print(f"\nrelaxed fallback: still missing {missing}")
            relaxed_candidates = _get_candidate_datasets(api, tarea, relaxed=True)
            _process_candidates(
                api=api,
                datasets=relaxed_candidates,
                task=tarea,
                target_per_level=target_per_level,
                datasets_por_nivel=datasets_por_nivel,
                processed_refs=processed_refs,
                allow_multi_csv=settings.auto.allow_multi_csv_on_relaxed,
                selected_levels=selected_levels,
                max_size=max_size,
                size_cache={},
            )

        missing = {
            nivel: target_per_level - count
            for nivel, count in datasets_por_nivel.items()
            if count < target_per_level
        }
        if missing:
            print(f"\nwarning: could not fill all levels for task '{tarea}'.")
            print(f"missing: {missing}")

    _print_summary()


def _get_candidate_datasets(api, task, relaxed=False):
    candidates = []
    seen_refs = set()

    suffixes = (
        settings.auto.relaxed_search_suffixes
        if relaxed
        else settings.auto.search_suffixes
    )
    max_pages = (
        settings.auto.relaxed_search_pages if relaxed else settings.auto.search_pages
    )

    for suffix in suffixes:
        query = f"{task}{suffix}".strip()
        for page in range(1, max_pages + 1):
            try:
                kwargs = {
                    "search": query,
                    "sort_by": settings.auto.sort_by,
                    "page": page,
                }
                if not relaxed:
                    kwargs.update(
                        {
                            "file_type": settings.auto.file_type,
                            "min_size": int(settings.auto.min_size),
                            "max_size": int(settings.auto.max_size),
                        }
                    )

                datasets = api.dataset_list(**kwargs)
            except TypeError:
                # older/newer SDK compatibility fallback when page is unsupported
                fallback_kwargs = {
                    "search": query,
                    "sort_by": settings.auto.sort_by,
                }
                if not relaxed:
                    fallback_kwargs.update(
                        {
                            "file_type": settings.auto.file_type,
                            "min_size": int(settings.auto.min_size),
                            "max_size": int(settings.auto.max_size),
                        }
                    )
                datasets = api.dataset_list(**fallback_kwargs)
                page = max_pages

            if not datasets:
                break

            new_in_page = 0
            for dataset in datasets:
                if dataset and dataset.ref not in seen_refs:
                    seen_refs.add(dataset.ref)
                    candidates.append(dataset)
                    new_in_page += 1

            if new_in_page == 0:
                break

    return candidates


def _process_candidates(
    api,
    datasets,
    task,
    target_per_level,
    datasets_por_nivel,
    processed_refs,
    allow_multi_csv,
    selected_levels,
    max_size,
    size_cache,
):
    for d in datasets:
        if all(count >= target_per_level for count in datasets_por_nivel.values()):
            print(f"\ncompleted: {target_per_level} datasets per level")
            break

        if d is None or d.ref in processed_refs:
            continue
        processed_refs.add(d.ref)

        print(f"\nanalyzing: {d.ref}")
        try:
            total_size = _get_dataset_total_size_bytes(api, d.ref, size_cache)
            if total_size is None:
                print("   skipping: could not determine dataset size")
                continue
            if total_size is not None and total_size > max_size:
                print(
                    f"   skipping: dataset size {total_size} exceeds max_size {max_size}"
                )
                continue

            temp_path = kagglehub.dataset_download(d.ref)

            is_single, csv_files = has_single_csv(temp_path)
            if not is_single and not allow_multi_csv:
                csv_count = len(csv_files)
                print(f"   skipping: {csv_count} csv files found, need exactly 1")
                continue

            csv_path = get_csv_path(temp_path)
            if csv_path is None and allow_multi_csv and csv_files:
                csv_path = _pick_csv_path(csv_files)
                print(
                    f"   relaxed fallback: selected one csv from {len(csv_files)} candidates"
                )

            if csv_path is None:
                print("   skipping: no usable csv found")
                continue

            df = read_dataset_sample(csv_path)

            metricas = analizar_dataframe(df, task)
            nivel, razones = determinar_nivel(metricas, task)

            if nivel not in selected_levels:
                print(f"   skipping: level '{nivel}' not selected")
                continue

            if datasets_por_nivel.get(nivel, 0) >= target_per_level:
                print(f"   skipping: already have enough '{nivel}' datasets")
                continue

            dataset_dir = os.path.join(os.getcwd(), "datasets", task, nivel)
            ensure_dir(dataset_dir)

            dataset_name = d.ref.replace("/", "_")
            final_path = os.path.join(dataset_dir, dataset_name)

            if os.path.exists(final_path):
                print("   skipping: dataset already exists in destination")
                continue

            copy_dataset(temp_path, final_path)

            datasets_por_nivel[nivel] = datasets_por_nivel.get(nivel, 0) + 1

            print_metrics(metricas, nivel, razones)
            print(f"   saved to: datasets/{task}/{nivel}/")
            print(f"   progress: {datasets_por_nivel}")

        except Exception as e:
            print(f"   error: {e}")


def _get_dataset_total_size_bytes(api, dataset_ref, size_cache):
    if dataset_ref in size_cache:
        return size_cache[dataset_ref]

    total_size = None
    try:
        files_response = api.dataset_list_files(dataset_ref)
        dataset_files = getattr(files_response, "files", None) or []
        total_size = 0
        for dataset_file in dataset_files:
            file_size = getattr(dataset_file, "total_bytes", None)
            if file_size is not None:
                total_size += int(file_size)
    except Exception:
        total_size = None

    size_cache[dataset_ref] = total_size
    return total_size


def _pick_csv_path(csv_files):
    return max(csv_files, key=lambda p: os.path.getsize(p))


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


def _count_existing_by_level(task, selected_levels):
    counts = dict.fromkeys(selected_levels, 0)

    for level in selected_levels:
        level_dir = os.path.join(os.getcwd(), "datasets", task, level)
        if not os.path.isdir(level_dir):
            continue

        dataset_dirs = [
            entry
            for entry in os.listdir(level_dir)
            if os.path.isdir(os.path.join(level_dir, entry))
        ]
        counts[level] = len(dataset_dirs)

    return counts
