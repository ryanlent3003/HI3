#!/usr/bin/env python3
"""Step 3: Generate analysis summary from model outputs.

This script reads output CSVs and writes a compact interpretation text file.
"""

from pathlib import Path
import pandas as pd


def main() -> None:
    outputs = Path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    perf_file = outputs / "performance_summary_by_site.csv"
    if not perf_file.exists():
        raise FileNotFoundError(
            "Missing outputs/performance_summary_by_site.csv. Run scripts/02_train_evaluate_lstm.py first."
        )

    perf = pd.read_csv(perf_file)
    test_row = perf.loc[perf["role"] == "test"].iloc[0]

    summary = [
        "CVEN 6920 Assignment 3 - Figure/Analysis Summary",
        "",
        f"Test site: {test_row['site_no']} ({test_row['site_name']})",
        f"NSE: {test_row['nse']:.3f}",
        f"KGE: {test_row['kge']:.3f}",
        f"RMSE (cfs): {test_row['rmse_cfs']:.3f}",
        f"MAE (cfs): {test_row['mae_cfs']:.3f}",
        "",
        "Key figures:",
        "- outputs/study_sites_map_conus_inset.png",
        "- outputs/training_history.png",
        "- outputs/obs_vs_pred_timeseries_09070500.png",
        "- outputs/obs_vs_pred_scatter_09070500.png",
    ]

    out_text = outputs / "analysis_summary.txt"
    out_text.write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote {out_text}")


if __name__ == "__main__":
    main()
