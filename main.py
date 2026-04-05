#!/usr/bin/env python3
"""
kaggle dataset downloader for automl testing

modes:
    auto: automatically search and download datasets by difficulty
    url: download datasets from urls listed in a text file
"""

import argparse
from auto_downloader import run_auto_collection
from config import settings, positive_int
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
        default=settings.url.default_url_file,
        help="path to file with dataset urls (for url mode, default: dataset_urls.txt)",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=settings.auto.valid_tasks,
        default=list(settings.auto.default_tasks),
        help="tasks to collect (for auto mode, default: classification regression)",
    )

    parser.add_argument(
        "--count",
        type=positive_int,
        default=settings.auto.default_count,
        help="datasets per difficulty level (for auto mode, default: 1)",
    )

    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=settings.auto.difficulty_levels,
        default=list(settings.auto.difficulty_levels),
        help="difficulty tiers to collect (for auto mode, default: easy mid hard)",
    )

    args = parser.parse_args()

    if args.mode == "auto":
        run_auto_collection(
            tasks=args.tasks,
            target_per_level=args.count,
            tiers=args.tiers,
        )
    elif args.mode == "url":
        run_url_download(url_file=args.url_file)


if __name__ == "__main__":
    main()
