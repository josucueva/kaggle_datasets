# Kaggle Dataset Downloader

Simple CLI project to download Kaggle datasets and organize them by:

- task: classification or regression
- difficulty: easy, mid, hard

## What This Project Does

- Auto mode: searches Kaggle and collects datasets by task and difficulty
- URL mode: downloads datasets from a custom URL/reference list
- Difficulty analyzer: classifies each dataset as easy, mid, or hard

All datasets are saved under `datasets/`.

## Folder Layout

```
datasets/
  classification/
    easy/
    mid/
    hard/
  regression/
    easy/
    mid/
    hard/
  urls/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure Kaggle API is configured (`~/.kaggle/kaggle.json`).

## Commands

### Auto Mode

Use auto mode to search and download by task and tier.

```bash
# default run
python main.py auto

# collect 5 datasets per tier
python main.py auto --count 5

# only easy tier
python main.py auto --tiers easy --count 10

# only classification
python main.py auto --tasks classification --count 5

# classification + regression, only easy and mid
python main.py auto --tasks classification regression --tiers easy mid --count 4
```

Notes:

- The downloader first uses strict search filters.
- If some tiers are still missing, it runs a relaxed fallback search.
- `max_size` in `config.py` is enforced as a hard size cap.

### URL Mode

Use URL mode to download datasets listed in a file.

```bash
# use dataset_urls.txt
python main.py url

# use a custom file
python main.py url --url-file my_urls.txt
```

Supported URL/reference formats:

- `username/dataset-name`
- `https://www.kaggle.com/datasets/username/dataset-name`
- `kaggle://username/dataset-name`

## URL File Format

Use section headers so downloads are classified by task:

```text
# classification
username/dataset-a
username/dataset-b

# regression
username/dataset-c
```

Behavior:

- Entries under `# classification` are analyzed and saved to:
  `datasets/classification/{easy,mid,hard}/...`
- Entries under `# regression` are analyzed and saved to:
  `datasets/regression/{easy,mid,hard}/...`
- Entries outside a section are saved to `datasets/urls/`.

## Difficulty Rules (Simple)

Easy is strict. A dataset is easy only if it has:

- numeric features only
- 15 columns or fewer
- 0 missing values
- 0 invalid values
- for classification: largest class ratio <= 0.60

If it fails easy rules, it is scored as mid or hard based on complexity signals
(missing data, invalid values, non-numeric ratio, dimensionality, imbalance, etc.).

## Main Files

- `main.py`: CLI entrypoint
- `auto_downloader.py`: auto collection logic
- `url_downloader.py`: URL list collection logic
- `dataset_analyzer.py`: difficulty analysis
- `config.py`: defaults and limits
- `dataset_urls.txt`: manual URL/reference list
