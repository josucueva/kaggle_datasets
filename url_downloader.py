import os
import kagglehub
from utils import has_single_csv, ensure_dir, copy_dataset


def read_urls_from_file(filepath):
    """read dataset urls from a text file"""
    if not os.path.exists(filepath):
        print(f"error: file '{filepath}' not found")
        return []

    with open(filepath, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

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

    # direct reference format
    if "/" in url and not url.startswith("http"):
        return url

    # kaggle:// protocol
    if url.startswith("kaggle://"):
        return url.replace("kaggle://", "")

    # full url
    if "kaggle.com/datasets/" in url:
        parts = url.split("kaggle.com/datasets/")
        if len(parts) > 1:
            return parts[1].rstrip("/")

    return None


def run_url_download(url_file="dataset_urls.txt"):
    """
    download datasets from urls listed in a text file

    args:
        url_file: path to text file containing dataset urls
    """
    print("kaggle url downloader")
    print(f"reading urls from: {url_file}\n")

    urls = read_urls_from_file(url_file)

    if not urls:
        print("no urls found in file")
        return

    print(f"found {len(urls)} url(s) to process\n")

    output_dir = os.path.join(os.getcwd(), "datasets", "urls")
    ensure_dir(output_dir)

    successful = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] processing: {url}")

        dataset_ref = parse_kaggle_url(url)

        if not dataset_ref:
            print("   error: invalid url format")
            failed += 1
            continue

        try:
            print(f"   downloading: {dataset_ref}")
            temp_path = kagglehub.dataset_download(dataset_ref)

            is_single, csv_files = has_single_csv(temp_path)

            if not is_single:
                print(f"   warning: {len(csv_files)} csv files found (expected 1)")

            dataset_name = dataset_ref.replace("/", "_")
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
