# Streamflow Prediction with LSTM (Upper Colorado Basin)

This repository contains my CVEEN 6920 Assignment 3 project for daily streamflow prediction using a Long Short-Term Memory (LSTM) model. The analysis combines USGS streamflow records with Daymet meteorological forcings and static site descriptors to build a regional model that is trained on multiple gauges and tested on an unseen gauge.

The workflow is designed to be reproducible end-to-end. Running the scripts in order downloads source data, prepares features, trains/evaluates the model, and generates the figures and summary outputs used in the report. Generated data and artifacts are intentionally excluded from version control, so the repository stays lightweight and focused on code, configuration, and documentation.

GitHub repository link (for report):
- https://github.com/ryanlent3003/HI3

## Project Figure
![Upper Colorado Site Map](docs/figures/readme_preview.png)

Figure: Study-site geographic context map used in the streamflow LSTM analysis.

## Repository Structure
```text
CVEN6920_Assignment3/
├── README.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── scripts/
│   ├── 01_data_acquisition_processing.py
│   ├── 02_train_evaluate_lstm.py
│   ├── 03_figures_analysis.py
│   └── run_lstm_upper_colorado.py
├── docs/
│   └── figures/
│       └── readme_preview.png
├── reports/
│   ├── manuscript_draft.md
│   └── assignment3_report_template.md
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
└── logs/
```

## What Each Script Does
- `scripts/01_data_acquisition_processing.py`: downloads/rebuilds site-level datasets and writes processed tables.
- `scripts/02_train_evaluate_lstm.py`: runs model training, validation, testing, and writes metrics/predictions/plots.
- `scripts/03_figures_analysis.py`: creates a compact analysis summary from model outputs.

## Environment Setup
You can use either `pip` or `conda`.

### Option A: pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate cven6920-assignment3
```

## Run Order
From the repository root:

```bash
python scripts/01_data_acquisition_processing.py
python scripts/02_train_evaluate_lstm.py
python scripts/03_figures_analysis.py
```

The scripts are non-interactive and create required folders automatically.

## Reproducibility Notes
- No raw/processed data or generated outputs are committed to Git.
- Data are retrieved programmatically (USGS NWIS + Daymet), so internet access is required for data download.
- The `.gitignore` keeps generated artifacts out of source control while retaining folder structure via `.gitkeep` files.
