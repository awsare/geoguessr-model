# GeoLocate

A machine learning project to classify images by geographic sector,
using the Kaggle dataset
[`ubitquitin/geolocation-geoguessr-images-50k`](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
(~50k GeoGuessr Street View images across ~124 countries).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python download_dataset.py   # download the dataset via kagglehub
python prepare_dataset.py    # build data/manifest.csv
python dataset.py            # sanity-check the PyTorch Dataset/DataLoader
python train.py              # train the CNN and save a checkpoint
python evaluate.py           # load the checkpoint and report test accuracy
python smoke_test.py         # quick end-to-end pipeline check on tiny data slices
```

## Data flow

`download_dataset.py` → `prepare_dataset.py` (using `sectors.py`) → `data/manifest.csv` → `dataset.py` → `train.py` → `checkpoints/geolocate_net.pth` → `evaluate.py`

## Files

- **`download_dataset.py`** — Downloads the dataset via `kagglehub`, reusing
  an existing cached download if present (`find_cached_download()`).

- **`sectors.py`** — Maps each country to a geographic sector so the model
  classifies by region instead of by individual country, keeping every
  country's images instead of dropping small ones. Two granularities are
  provided (`continent`, `subregion`); `SECTOR_GRANULARITY` picks the active
  one.

- **`prepare_dataset.py`** — Builds `data/manifest.csv`: groups countries
  into sectors via `sectors.py`, drops sectors with too few images
  (`MIN_IMAGES_PER_SECTOR`), and stratifies each sector into train/val/test
  splits. `filepath` points directly into the kagglehub cache, so
  `manifest.csv` isn't portable across machines without re-running this
  script, so it stays local and should be regenerated per machine.

- **`dataset.py`** — `GeoLocateDataset`, a `torch.utils.data.Dataset` that
  reads the manifest and loads/transforms images (224x224, ImageNet
  normalization). Sector labels are encoded via a mapping persisted to
  `data/label_map.json`.

- **`train.py`** — Trains a from-scratch CNN (`Net`) with `CrossEntropyLoss`
  + SGD and saves `checkpoints/geolocate_net.pth`.

- **`evaluate.py`** — Loads `checkpoints/geolocate_net.pth` and reports
  overall and per-sector test accuracy for the test split.

- **`smoke_test.py`** — Exercises the full pipeline (dataset → model →
  training step → checkpoint → eval) on a tiny data slice, to catch
  breakage quickly without a full training run.

- **`exploration.ipynb`** — Dataset exploration of per-country image
  counts and class imbalance.

- **`data/`** — Gitignored except for `data/label_map.json`. The generated
  `manifest.csv` stays local because it contains machine-specific kagglehub
  cache paths. Images stay in the kagglehub cache, not in this repo.
