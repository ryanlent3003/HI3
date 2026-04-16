#!/usr/bin/env python3
"""Step 1: Data acquisition and processing for CVEN 6920 Assignment 3.

This script downloads/rebuilds site-level hydrologic dataframes and saves them.
It runs with no user input and creates required folders automatically.
"""

from pathlib import Path

from run_lstm_upper_colorado import Config, SiteConfig, ensure_dir, fetch_site_info, build_site_dataframe


def main() -> None:
    cfg = Config()
    # Keep generated data out of git-tracked source; .gitignore handles this folder.
    processed_dir = Path("data") / "processed"
    ensure_dir(processed_dir)

    for site in cfg.sites:
        # Retrieve metadata and build merged dataframe (USGS + Daymet + static attrs).
        meta = fetch_site_info(site.site_no)
        df = build_site_dataframe(site, cfg, meta=meta)

        out_file = processed_dir / f"site_dataframe_{site.site_no}.csv"
        df.reset_index().rename(columns={"index": "date"}).to_csv(out_file, index=False)
        print(f"Saved {out_file} ({len(df)} rows)")

    print("Data acquisition and processing complete.")


if __name__ == "__main__":
    main()
