# -*- coding: utf-8 -*-
"""Download MODIS data to CX1 Ephemeral storage."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import check_call

from tqdm import tqdm

if __name__ == "__main__":
    # CX1 specific setup.
    data_dir = str(Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006")

    # Environment bin path.
    env_bin = Path(
        "/rds/general/user/ahk114/home/.pyenv/versions/miniconda3-latest/envs/"
        "wildfires/bin"
    )
    modis_download_bin = str(env_bin / "modis_download.py")

    def download_modis_day(day_str):
        """Download a single day."""
        return check_call(
            [
                modis_download_bin,
                "-p",
                "MOD15A2H.006",
                "-f",
                day_str,
                "-O",
                "-r",
                data_dir,
            ]
        )

    futures = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        for year in range(2012, 1999, -1):
            # Start on the first day of each year.
            date = datetime(year, 1, 1)
            for days in range(0, 365, 8):
                # Advance by 8 days.
                date += timedelta(days=days)
                date_str = date.strftime("%Y-%m-%d")
                futures.append(executor.submit(download_modis_day, date_str))

        # Progress bar.
        for future in tqdm(
            as_completed(futures), desc="Downloading MODIS data", total=len(futures)
        ):
            future.result()
