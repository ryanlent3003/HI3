# Hydroinformatics Assignment #3: Machine Learning Applications

## Project Track
This repository implements the **Streamflow Prediction using LSTM Models** track for CVEN 6920 Assignment #3.

## Repository Purpose
This project demonstrates an end-to-end, reproducible machine learning workflow in Python for hydrologic streamflow prediction, including:
- Data acquisition from public sources
- Data processing and feature engineering
- Model training and evaluation
- Figure generation and interpretation support

## GitHub Link (for report)
Live repository URL:
- https://github.com/ryanlent3003/HI3

## Repository Organization
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
├── notebooks/
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

## README Figure
![Upper Colorado Site Map](docs/figures/readme_preview.png)

*Figure: Study-site geographic context map used in the streamflow LSTM analysis.*

## Reproducible Setup
You can run this project with either `pip` or `conda`.

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

## Reproducible Execution (No User Input Required)
Run from repository root. Scripts are designed to create required folders and run without interactive prompts.

```bash
python scripts/01_data_acquisition_processing.py
python scripts/02_train_evaluate_lstm.py
python scripts/03_figures_analysis.py
```

## Script Roles
- `scripts/01_data_acquisition_processing.py`
  - Retrieves and merges hydrologic datasets (USGS + Daymet + static metadata).
  - Writes processed site-level data tables.
- `scripts/02_train_evaluate_lstm.py`
  - Runs the full LSTM training/evaluation pipeline.
  - Produces metrics, predictions, and primary figures.
- `scripts/03_figures_analysis.py`
  - Summarizes core output metrics and figure references for reporting.

## Data Policy (No Data in Repository)
This repository intentionally excludes generated data and outputs from version control to satisfy assignment requirements. The `.gitignore` file excludes:
- `data/raw/*`
- `data/processed/*`
- `outputs/*`
- `logs/*`

Only source code, configuration, templates, and lightweight documentation assets are tracked.

## Assignment Deliverables Coverage
- GitHub-ready project layout with `.gitignore`: **included**
- Well-organized Python scripts and reproducible workflow: **included**
- Separate scripts for acquisition/processing and analysis/figures: **included**
- README with usage instructions and image: **included**
- Virtual environment/dependency file: **included** (`environment.yml`, `requirements.txt`)
- No committed data products: **configured via `.gitignore`**
- Report support with required section template and GitHub link field: **included** (`reports/assignment3_report_template.md`)

## Report Preparation Notes
Use `reports/assignment3_report_template.md` as the structure for the final report PDF. It includes the required sections:
- Cover Page
- Project Summary
- Background
- Methods
- Results
- Discussion
- Conclusion
- References

Formatting reminders (per assignment prompt):
- Maximum 5 pages (excluding cover page and references)
- Single-spaced, 12 pt Calibri (or similar), 1-inch margins
- Export as PDF for submission

## Commit/Push Requirement Reminder
The rubric requires multiple commits/pushes. Suggested commit sequence:
1. Initial scaffold (`README`, env files, `.gitignore`)
2. Scripts and reproducibility updates
3. Report and manuscript files
4. Final cleanup and validation
