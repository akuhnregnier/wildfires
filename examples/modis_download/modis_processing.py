# -*- coding: utf-8 -*-
import logging
import shutil
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from time import sleep

from configuration import (
    data_dir,
    download_file,
    env_bin,
    output_dir,
    processed_file,
    temporary_dir,
)

from wildfires.data.mosaic_modis_tiles import get_file_dates, mosaic_process_date
from wildfires.logging_config import enable_logging

gdal_translate = env_bin / "gdal_translate"
gdalwarp = env_bin / "gdalwarp"

enable_logging(level="INFO")
logger = logging.getLogger(__name__)


def run(queue, data_dir):
    """Process the MODIS data in `data_dir`.

    Returns None if an error occurred, and the processed date otherwise.

    """

    def handle_error():
        queue.put(None)

    file_map = get_file_dates(data_dir)

    if len(file_map) != 1:
        logger.error(f"File map had length '{len(file_map)}' for dir: {data_dir}.")
        return handle_error()

    date, date_files = next(iter(file_map.items()))

    error = False
    try:
        mosaic_process_date(
            date,
            date_files,
            temporary_dir,
            output_dir,
            memory=4000,
            multi=True,
            overwrite=True,
            gdal_translate=gdal_translate,
            gdalwarp=gdalwarp,
        )
    except:
        logger.exception(f"Processing of '{date}' failed.")
        error = True
    finally:
        # Clean temporary dir.
        # NOTE: This makes this code single-threaded!!
        if temporary_dir.is_dir():
            shutil.rmtree(temporary_dir)
            temporary_dir.mkdir()

    if error:
        return handle_error()

    # Record this date as having been processed.
    year = int(date[:4])
    days = int(date[4:])
    queue.put(datetime(year, 1, 1) + timedelta(days=days - 1))


if __name__ == "__main__":
    # Continuously monitor the file recording any downloaded files and process any
    # previously unprocessed files accordingly.

    while True:
        logger.info("Checking for downloaded and unprocessed files")
        with download_file.open("r") as f:
            downloaded = f.read().strip().split("\n")
        with processed_file.open("r") as f:
            processed = f.read().strip().split("\n")

        outstanding = set(downloaded) - set(processed)
        logger.info(f"Outstanding dates: {outstanding}")
        for date_str in outstanding:
            logger.info(f"Processing: {date_str}")
            date_dir = data_dir / date_str

            # Carry out processing using a new process to avoid potential memory leaks.
            queue = Queue()
            p = Process(target=run, args=(queue, date_dir))
            p.start()
            processed_date = queue.get()
            p.join()

            if processed_date is not None and (
                f"{processed_date:%Y.%m.%d}" == date_str
            ):
                logger.info(f"Processed date: {date_str}")
                with processed_file.open("a") as f:
                    f.write(date_str + "\n")

                # Remove the original data directory.
                shutil.rmtree(date_dir)
            else:
                logger.error(f"Error during processing of date: {date_str}.")

        sleep(100)
