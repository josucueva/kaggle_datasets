import os
import re
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
from config import settings
from dataset_analyzer import analizar_dataframe, determinar_nivel, print_metrics
from utils import (
    has_single_csv,
    ensure_dir,
    copy_dataset,
    get_csv_path,
    read_dataset_sample,
)


def read_urls_from_file(filepath):
    """read dataset urls from a text file"""
    if not os.path.exists(filepath):
        print(f"error: file '{filepath}' not found")
        return []

    with open(filepath, "r") as f:
        urls = []
        current_task = None
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                line_l = line.lower()
                if "classification" in line_l:
                    current_task = "classification"
                elif "regression" in line_l:
                    current_task = "regression"
                continue

            urls.append({"url": line, "task": current_task})

    return urls


def parse_kaggle_url(url):
    """
    extract dataset reference from kaggle url
    examples:
        https://www.kaggle.com/datasets/username/dataset-name -> username/dataset-name
        kaggle://username/dataset-name -> username/dataset-name
        username/dataset-name -> username/dataset-name
    """
    url = url.strip()

    if url.startswith(settings.url.kaggle_protocol_prefix):
        dataset_ref = url[len(settings.url.kaggle_protocol_prefix) :]
    elif "kaggle.com/datasets/" in url:
        parts = url.split("kaggle.com/datasets/", maxsplit=1)
        if len(parts) <= 1:
            return None
        dataset_ref = parts[1].split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    elif "/" in url and not url.startswith("http"):
        dataset_ref = url
    else:
        return None

    dataset_ref = dataset_ref.strip().strip("/")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9._-]*", dataset_ref
    ):
        return None

    return dataset_ref


def run_url_download(url_file=settings.url.default_url_file):
    """
    download datasets from urls listed in a text file

    args:
        url_file: path to text file containing dataset urls
    """
    print("kaggle url downloader")
    print(f"reading urls from: {url_file}\n")
    max_size = int(settings.auto.max_size)
    print(f"max dataset size cap: {max_size} bytes")

    api = KaggleApi()
    api.authenticate()
    size_cache = {}

    urls = read_urls_from_file(url_file)

    if not urls:
        print("no urls found in file")
        return

    print(f"found {len(urls)} url(s) to process\n")

    output_dir = os.path.join(os.getcwd(), "datasets", "urls")
    ensure_dir(output_dir)

    successful = 0
    failed = 0

    for i, item in enumerate(urls, 1):
        url = item["url"]
        task = item["task"]

        print(f"\n[{i}/{len(urls)}] processing: {url}")

        dataset_ref = parse_kaggle_url(url)

        if not dataset_ref:
            print("   error: invalid url format")
            failed += 1
            continue

        try:
            total_size = _get_dataset_total_size_bytes(api, dataset_ref, size_cache)
            if total_size is None:
                print("   skipping: could not determine dataset size")
                failed += 1
                continue
            if total_size is not None and total_size > max_size:
                print(
                    f"   skipping: dataset size {total_size} exceeds max_size {max_size}"
                )
                failed += 1
                continue

            print(f"   downloading: {dataset_ref}")
            temp_path = kagglehub.dataset_download(dataset_ref)

            is_single, csv_files = has_single_csv(temp_path)

            if not is_single:
                print(f"   warning: {len(csv_files)} csv files found (expected 1)")

            dataset_name = dataset_ref.replace("/", "_")

            if task and is_single:
                csv_path = get_csv_path(temp_path)
                df = read_dataset_sample(csv_path)
                metricas = analizar_dataframe(df, task)
                nivel, razones = determinar_nivel(metricas, task)

                final_dir = os.path.join(os.getcwd(), "datasets", task, nivel)
                ensure_dir(final_dir)
                final_path = os.path.join(final_dir, dataset_name)
                copy_dataset(temp_path, final_path)

                print_metrics(metricas, nivel, razones)
                print(f"   task: {task}")
                print(f"   saved to: datasets/{task}/{nivel}/{dataset_name}/")
            else:
                if not task:
                    print(
                        "   warning: task not specified in url file, saving as unclassified"
                    )
                final_path = os.path.join(output_dir, dataset_name)
                copy_dataset(temp_path, final_path)
                print(f"   saved to: datasets/urls/{dataset_name}/")

            successful += 1

        except Exception as e:
            error_msg = str(e)
            print(f"   error: {error_msg}")

            # provide helpful guidance for common errors
            if "403" in error_msg or "permission" in error_msg.lower():
                print("   hint: dataset may be private or require terms acceptance")
                print(f"   visit: https://www.kaggle.com/datasets/{dataset_ref}")
                print("   - verify the dataset exists and is public")
                print("   - accept any terms/conditions on the website first")
            elif "404" in error_msg:
                print("   hint: dataset not found - check the username/dataset-name")
                print(f"   visit: https://www.kaggle.com/datasets/{dataset_ref}")

            failed += 1

    _print_summary(successful, failed)


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


def _print_summary(successful, failed):
    """print download summary"""
    print(f"\n{'='*50}")
    print("summary")
    print(f"{'='*50}")
    print(f"\nsuccessful: {successful}")
    print(f"failed: {failed}")
    print(f"total: {successful + failed}")
    print("\ndatasets saved to: datasets/urls/")

    if failed > 0:
        print("\ncommon issues:")
        print("  - dataset requires terms acceptance (visit kaggle.com first)")
        print("  - dataset is private or doesn't exist")
        print("  - incorrect username/dataset-name format")

    print()
