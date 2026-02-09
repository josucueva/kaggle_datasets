# kaggle dataset downloader

modular dataset collection tool for automl testing

## structure

```
.
├── main.py              # entry point with cli
├── kaggle_downloader.py     # auto search & download by difficulty
├── url_downloader.py        # download from url list
├── dataset_analyzer.py      # dataset difficulty analysis
├── utils.py                 # shared utilities
├── dataset_urls.txt         # url list for manual mode
└── datasets/                # output directory
    ├── classification/
    │   ├── easy/
    │   ├── mid/
    │   └── hard/
    ├── regression/
    │   ├── easy/
    │   ├── mid/
    │   └── hard/
    └── urls/                # manually specified datasets
```

## usage

### auto mode

automatically search and download datasets organized by difficulty:

```bash
# basic usage (1 dataset per difficulty level)
python main.py auto

# download 3 datasets per level
python main.py auto --count 3

# only classification datasets
python main.py auto --tasks classification

# both tasks, 5 datasets each
python main.py auto --tasks classification regression --count 5
```

### url mode

download specific datasets from a text file:

```bash
# use default file (dataset_urls.txt)
python main.py url

# use custom file
python main.py url --url-file my_datasets.txt
```

### url file format

create a text file with one url per line:

```
# comments start with #
username/dataset-name
https://www.kaggle.com/datasets/username/another-dataset
kaggle://username/third-dataset
```

## modules

- **main.py**: cli interface, choose mode and parameters
- **kaggle_downloader.py**: auto mode - search by task type and difficulty
- **url_downloader.py**: url mode - download from specified urls
- **dataset_analyzer.py**: analyze datasets and classify difficulty
- **utils.py**: file operations and helpers

## difficulty classification

datasets are classified based on:

- missing data ratio
- numeric vs categorical features
- dimensionality (features/samples)
- cardinality of categorical features
- class imbalance (classification)
- number of classes (classification)

scoring:

- easy: 0-2 points
- mid: 3-5 points
- hard: 6+ points

## requirements

see requirements.txt
