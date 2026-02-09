#!/usr/bin/env python3
"""
kaggle dataset downloader for automl testing

modes:
    auto: automatically search and download datasets by difficulty
    url: download datasets from urls listed in a text file
"""

import argparse
from auto_downloader import run_auto_collection
from url_downloader import run_url_download


def main():
    parser = argparse.ArgumentParser(
        description="kaggle dataset downloader for automl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mode",
        choices=["auto", "url"],
        help="download mode: auto (search by difficulty) or url (from file)",
    )

    parser.add_argument(
        "--url-file",
        default="dataset_urls.txt",
        help="path to file with dataset urls (for url mode, default: dataset_urls.txt)",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["classification", "regression"],
        help="tasks to collect (for auto mode, default: classification regression)",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="datasets per difficulty level (for auto mode, default: 1)",
    )

    args = parser.parse_args()

    if args.mode == "auto":
        run_auto_collection(tasks=args.tasks, target_per_level=args.count)
    elif args.mode == "url":
        run_url_download(url_file=args.url_file)


if __name__ == "__main__":
    main()
