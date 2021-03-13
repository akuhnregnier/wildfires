# -*- coding: utf-8 -*-
"""Authorization will be performed using netrc by requests automatically."""
import logging
import os
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from configuration import data_dir, download_file, processed_file, temporary_data_dir
from pymodis.downmodis import modisHtmlParser
from tqdm import tqdm

from wildfires.logging_config import enable_logging

enable_logging(level="INFO")
logger = logging.getLogger(__name__)

base_url = "https://e4ftl01.cr.usgs.gov/MOLT/MOD15A2H.006/"


def urljoin(*args):
    return "/".join(args).replace("//", "/").replace(":/", "://")


def worker_log(log_file, msg):
    with log_file.open("a") as f:
        f.write(f"{datetime.now()}: {msg}" + "\n")


def safe_write(content, filename):
    """Write content to a temporary file first then move this."""
    tmp_file = temporary_data_dir / (
        f"{filename.name}-{id(threading.current_thread())}-{os.getpid()}"
    )
    # Write.
    with tmp_file.open("wb") as f:
        f.write(content)
    # Then move.
    shutil.move(tmp_file, filename)


def get_all_dates():
    return modisHtmlParser(requests.get(base_url).content).get_dates()


def get_hdf_files(date):
    return [
        f
        for f in modisHtmlParser(
            requests.get(urljoin(base_url, date)).content
        ).get_all()
        if f.endswith(".hdf")
    ]


def download_date(date):
    dest_dir = Path(data_dir) / date
    dest_dir.mkdir(exist_ok=True, parents=True)

    log_file = dest_dir / "worker_log.txt"

    def write_worker_log(msg):
        worker_log(log_file, msg)

    def remove_empty():
        # Manually remove empty files.
        for f in dest_dir.glob("*"):
            if f.stat().st_size == 0:
                # Remove empty files.
                f.unlink()

    hdf_files = get_hdf_files(date)
    write_worker_log(f"{len(hdf_files)} HDF files for date: {date}.")

    session = requests.Session()

    # Repeated tries in case something goes wrong.
    for i in range(30):
        try:
            remove_empty()

            # Ignore already existing files.
            existing_files = [f.name for f in list(dest_dir.glob("*.hdf"))]
            filtered_hdf_files = [f for f in hdf_files if f not in existing_files]

            # Download the remaining files.
            write_worker_log(f"Downloading {len(filtered_hdf_files)} new files.")

            for hdf_file in filtered_hdf_files:
                dest_file = dest_dir / hdf_file
                file_url = urljoin(base_url, date, hdf_file)

                write_worker_log(f"Downloading file {hdf_file} from {file_url}.")

                content = session.get(file_url).content

                safe_write(content, dest_file)
                write_worker_log(f"Finished downloading file {hdf_file}.")

            # Break out of the loop on success.
            break
        except Exception as e:
            write_worker_log(e)
    else:
        return

    if len(list(dest_dir.glob("*.hdf"))) != len(hdf_files):
        return

    return len(hdf_files)


if __name__ == "__main__":
    all_dates = get_all_dates()

    workers = 15

    with download_file.open("r") as f:
        already_downloaded = f.read().strip().split("\n")

    with processed_file.open("r") as f:
        already_processed = f.read().strip().split("\n")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_date = {
            executor.submit(download_date, date): date
            for date in all_dates
            if date not in (already_downloaded + already_processed)
        }

        for f in tqdm(
            as_completed(future_to_date),
            desc="Downloading MODIS",
            total=len(future_to_date),
            smoothing=0,
        ):
            if not f.result():
                logger.error(f"Error for date: {future_to_date[f]}")
            else:
                logger.info(f"Done downloading: {future_to_date[f]}")
                with download_file.open("a") as download_f:
                    download_f.write(future_to_date[f] + "\n")
