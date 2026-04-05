# kaggle dataset downloader

modular dataset collection tool for automl testing

## structure

```
.
├── main.py              # entry point with cli
├── auto_downloader.py       # auto search & download by difficulty
├── url_downloader.py        # download from url list
├── dataset_analyzer.py      # dataset difficulty analysis
├── config.py                # centralized configuration
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

# only easy datasets (both tasks)
python main.py auto --tiers easy --count 10

# both tasks, 5 datasets each
python main.py auto --tasks classification regression --count 5
```

auto mode now uses a two-pass strategy to improve tier fill rates:

- pass 1: strict search (csv + size filters + single-csv datasets)
- pass 2: relaxed fallback (broader search keywords/pages; can select one csv from multi-csv datasets)

this helps when strict easy/hard buckets are hard to fill with the initial candidate pool.

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
# use section headers to classify url downloads by task:
# classification
username/dataset-name
# regression
https://www.kaggle.com/datasets/username/another-dataset
kaggle://username/third-dataset
```

when url mode finds `# classification` or `# regression` headers, downloaded datasets are analyzed and saved to:

- `datasets/classification/{easy,mid,hard}/...`
- `datasets/regression/{easy,mid,hard}/...`

urls outside those sections are saved as unclassified in `datasets/urls/`.

### auto-sklearn baseline mode (thesis comparison)

run default auto-sklearn over all downloaded tiered datasets, with task/tier filters.

required environment setup for auto-sklearn baseline compatibility:

```bash
# create dedicated python 3.9 env
uv venv -p 3.9 .venv-autosklearn
source .venv-autosklearn/bin/activate

# provide swig binary for pyrfr build
uv pip install swig

# install project deps (auto-sklearn pinned)
uv pip install -r requirements.txt
```

examples:

- run all tasks and tiers:
  python -m autosklearn_baseline.cli
- run only classification mid tier:
  python -m autosklearn_baseline.cli --task classification --tier mid
- run only regression easy and hard tiers:
  python -m autosklearn_baseline.cli --task regression --tier easy hard

optional runtime controls:

- `--time-left`: total time budget per dataset (seconds)
- `--per-run-time-limit`: time budget per model fit candidate (seconds)
- `--test-size`: train/test split ratio
- `--seed`: random seed
- `--target-config`: JSON map with explicit target column per dataset

explicit target mapping file:

- `autosklearn_baseline/dataset_targets.json`

the runner uses this mapping first to select the target column for each dataset,
and only falls back to heuristic inference if a dataset is not mapped.

outputs are saved in `results/`:

- `autosklearn_baseline_results.csv` (cumulative per-dataset outcomes, appended each run)
- `autosklearn_baseline_summary.json` (cumulative success rate and mean metrics)

the JSON summary also includes `datasets_latest`, with one latest entry per dataset
(task, tier, name), including status, key metrics, and `runs_count`.

the results CSV also stores model-traceability metadata per dataset, including:

- `automl_models_tried`
- `automl_ensemble_size`
- `automl_best_model_weight`
- `automl_best_model_repr`
- `automl_sprint_stats`
- `automl_show_models`

## modules

- **main.py**: cli interface, choose mode and parameters
- **auto_downloader.py**: auto mode - search by task type and difficulty
- **url_downloader.py**: url mode - download from specified urls
- **dataset_analyzer.py**: analyze datasets and classify difficulty
- **config.py**: central place for defaults and validation
- **utils.py**: file operations and helpers

## difficulty classification

the classifier now uses **strict easy rules** and a **point-based mid/hard score**.

### strict easy requirements

a dataset is marked as `easy` only if all these are true:

- all feature columns are numeric
- very few columns (`<= 15`)
- missing data ratio is exactly `0.0`
- invalid/corrupt value ratio is exactly `0.0`
- for classification: class imbalance is low (largest class ratio `<= 0.60`)

if any strict condition fails, the dataset cannot be `easy`.

### mid/hard scoring (when strict easy fails)

the analyzer adds points for complexity signals:

| Signal                      | Thresholds        | Points      |
| --------------------------- | ----------------- | ----------- |
| Missing data                | `>5%` / `>20%`    | `+1` / `+3` |
| Invalid values              | `>1%` / `>5%`     | `+1` / `+3` |
| Numeric feature ratio       | `<80%` / `<50%`   | `+1` / `+3` |
| Number of columns           | `>40` / `>100`    | `+1` / `+3` |
| Features/samples ratio      | `>0.10` / `>0.20` | `+1` / `+2` |
| Avg categorical cardinality | `>80` / `>200`    | `+1` / `+2` |

classification-only signals:

| Signal                                | Thresholds                  | Points             |
| ------------------------------------- | --------------------------- | ------------------ |
| Target not detected confidently       | n/a                         | `+2`               |
| Class imbalance (largest class ratio) | `>0.65` / `>0.75` / `>0.90` | `+1` / `+2` / `+3` |
| Number of classes                     | `>10` / `>20`               | `+1` / `+2`        |

regression-only signals:

| Signal                           | Thresholds | Points |
| -------------------------------- | ---------- | ------ |
| Target not detected confidently  | n/a        | `+1`   |
| Low numeric ratio for regression | `<50%`     | `+2`   |

final level:

- `easy`: passes all strict easy requirements
- `mid`: strict easy failed and score `< 6`
- `hard`: strict easy failed and score `>= 6`

## requirements

see requirements.txt
