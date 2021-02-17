# -*- coding: utf-8 -*-
"""Combine individual tiles into a large HDF file and regrid this.

MODIS LAI/fPAR files have the following bands (sub-datasets):

'HDF4_EOS:EOS_GRID:/rds/general/user/ahk114/home/scratch/modis.hdf:MOD_Grid_MOD15A2H:Fpar_500m'
'HDF4_EOS:EOS_GRID:/rds/general/user/ahk114/home/scratch/modis.hdf:MOD_Grid_MOD15A2H:Lai_500m'
'HDF4_EOS:EOS_GRID:/rds/general/user/ahk114/home/scratch/modis.hdf:MOD_Grid_MOD15A2H:FparLai_QC'
'HDF4_EOS:EOS_GRID:/rds/general/user/ahk114/home/scratch/modis.hdf:MOD_Grid_MOD15A2H:FparExtra_QC'
'HDF4_EOS:EOS_GRID:/rds/general/user/ahk114/home/scratch/modis.hdf:MOD_Grid_MOD15A2H:FparStdDev_500m'
'HDF4_EOS:EOS_GRID:/rds/general/user/ahk114/home/scratch/modis.hdf:MOD_Grid_MOD15A2H:LaiStdDev_500m'


E.g. as reported by
# >>> with rasterio.open(data_file) as dataset:
# >>>     print(dataset.subdatasets)

There are max 36 (lon, x) by 18 (lat, y) tiles.
Each tile is 2400 x 2400 pixels.

# x_pixels = 2400 * 36
# y_pixels = 2400 * 18

"""
import logging
import os
import re
import shlex
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path
from subprocess import check_output

import dask
import dask.array as da
import iris
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

from ..logging_config import enable_logging

logger = logging.getLogger(__name__)

tile_shape = (2400, 2400)
fill_value = 255  # Recognised as NODATA.
max_valid = 100  # Maximum valid data.

min_n_tiles = 200

fpar_band_name = "Fpar_500m"
qc_band_name = "FparLai_QC"

modis_proj = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
    " +no_defs"
)

bound_axes = {"NORTH": "y", "EAST": "x", "SOUTH": "y", "WEST": "x"}


@dask.delayed(pure=True)
def delayed_read_band_data(fpar_dataset_name, qc_dataset_name):
    """Read band data from a HDF4 file.

    Assumes the first dimensions have a size 1.

    FparLai_QC.

    Bit no. 5-7 3-4 2 1 0

    Acceptable values:
            000 00  0 0 0
            001 01  0 0 0

    Unacceptable mask:
            110 10  1 1 1

    """
    with rasterio.open(fpar_dataset_name) as dataset:
        fpar_data = dataset.read()[0]
    with rasterio.open(qc_dataset_name) as dataset:
        qc_data = dataset.read()[0]

    assert fpar_data.shape == tile_shape
    assert qc_data.shape == tile_shape

    # Ignore invalid and poor quality data.
    fpar_data[
        np.logical_or(fpar_data > max_valid, np.bitwise_and(qc_data, 0b11010111))
    ] = fill_value
    return fpar_data


def mosaic_process_date(
    date,
    date_files,
    temporary_dir,
    output_dir,
    memory,
    multi=False,
    overwrite=False,
    gdal_translate="gdal_translate",
    gdalwarp="gdalwarp",
):
    """Mosaic and regrid MODIS Fpar data from a given date.

    Args:
        date (str): MODIS date string, e.g. '2021034'.
        date_files (iterable of pathlib.Path): Files containing data for `date`.
        temporary_dir (pathlib.Path): Directory for temporary files.
        output_dir (pathlib.Path): Directory for output files.
        memory (int): GDAL memory in MB. Capped at 9999 MB.
        multi (bool): If True, add the '-multi' option to gdalwarp.
        overwrite (bool): If True, overwrite existing files.
        gdal_translate (str): gdal_translate command path.
        gdalwarp (str): gdalwarp command path.

    Returns:
        None or pathlib.Path: None if no processing could be done, or the filename of
            the processed data.

    """
    if len(date_files) < min_n_tiles:
        logger.warning(
            f"Found {len(date_files)} files (tiles) for '{date}'. "
            f"Expected at least {min_n_tiles}."
        )
        return None

    # Limit to 9999 because otherwise the parameter is interpreted as bytes instead of
    # megabytes.
    memory = min(9999, memory)

    output_base = temporary_dir / f"{fpar_band_name}_{date}"

    mosaic_file = output_base.with_name(output_base.stem + "_mosaic.hdf5")
    mosaic_vrt_file = mosaic_file.with_suffix(".vrt")
    regridded_file = output_base.with_name(output_base.stem + "_0d25_raw.nc")
    output_file = Path(output_dir) / (output_base.stem + "_0d25.nc")

    # Used to convert the bounding coordinates to MODIS (m) coordinates.
    # NOTE: transformer.transform(lat, lon) -> (x, y)
    transformer = Transformer.from_crs("EPSG:4326", modis_proj)

    bounds_coords = defaultdict(list)

    # Collection of 'delayed' objects containing the data, indexed using
    # (horizontal, vertical) MODIS tile numbers.
    tile_data = {}
    for data_file in date_files:
        fpar_dataset_name = (
            f"HDF4_EOS:EOS_GRID:{data_file}:MOD_Grid_MOD15A2H:{fpar_band_name}"
        )
        qc_dataset_name = (
            f"HDF4_EOS:EOS_GRID:{data_file}:MOD_Grid_MOD15A2H:{qc_band_name}"
        )
        with rasterio.open(fpar_dataset_name) as dataset:
            tags = dataset.tags()
            for bound_name, axis in bound_axes.items():
                bound_value = float(tags[f"{bound_name}BOUNDINGCOORDINATE"])
                bounds_coords[axis].append(bound_value)

        tile_data[
            tuple(
                # Parse the horizontal (h) and vertical (v) tile numbers.
                map(int, re.search(r"h(\d{2})v(\d{2})", str(data_file)).groups())
            )
        ] = da.from_delayed(
            delayed_read_band_data(fpar_dataset_name, qc_dataset_name),
            shape=tile_shape,
            dtype=np.uint8,
        )

    # Get the extreme bounding values in lat lon coordinates.
    extreme_bounds = {
        axis: (min(axis_bounds), max(axis_bounds))
        for axis, axis_bounds in bounds_coords.items()
    }
    logger.debug(f"{date} {extreme_bounds}")

    # Transform the extreme bounding values to MODIS coordinates for reprojection.
    modis_bounds = {}
    for axis, bounds in extreme_bounds.items():
        modis_bounds[axis] = sorted(
            transformer.transform(
                *(0, extreme_coord)[slice(None, None, 1 if axis == "x" else -1)]
            )[0 if axis == "x" else 1]
            for extreme_coord in bounds
        )

    logger.debug(f"{date} {modis_bounds}")

    # Create the mosaic of MODIS tiles.

    # Extract all possible vertical and horizontal tile numbers.
    hs, vs = zip(*tile_data)

    data_blocks = []

    # Iterate over all tiles, using existing data where possible.
    for v_index in range(min(vs), max(vs) + 1):
        data_blocks.append([])
        for h_index in range(min(hs), max(hs) + 1):
            data_blocks[-1].append(
                tile_data.get(
                    (h_index, v_index),
                    # Use full() to pad irrelevant tiles with the invalid data marker.
                    da.full(
                        tile_shape,
                        fill_value=fill_value,
                        dtype=np.uint8,
                        # XXX: Specifying 'chunksize' here causes the following error
                        # when calling 'to_hdf5':
                        # OSError: Can't write data (no appropriate function for conversion path)
                        # chunksize=tile_shape,
                    ),
                )
            )

    data = da.block(data_blocks)[::-1]

    if mosaic_file.is_file() and overwrite:
        logger.info(f"'{mosaic_file}' exists. Deleting.")
        mosaic_file.unlink()

    recalculate = False
    if not mosaic_file.is_file():
        recalculate = True
        data.to_hdf5(str(mosaic_file), "/fpar")
    else:
        logger.warning(f"'{mosaic_file}' exists. Not deleting.")

    # Attach information about the transform prior to calling 'gdalwarp'.
    y_pixels_max = data.shape[0] - 1
    x_pixels_max = data.shape[1] - 1

    y_min, y_max = modis_bounds["y"]
    x_min, x_max = modis_bounds["x"]

    gcp_opts = []
    for y_pixel, y_loc, x_pixel, x_loc in [
        (0, y_min, 0, x_min),
        (y_pixels_max, y_max, 0, x_min),
        (y_pixels_max, y_max, x_pixels_max, x_max),
        (0, y_min, x_pixels_max, x_max),
    ]:
        # -gcp <pixel> <line> <easting> <northing>
        gcp_opts.append(f"-gcp {x_pixel} {y_pixel} {x_loc} {y_loc}")

    cmd = " ".join(
        (
            f"{gdal_translate} -of VRT -a_srs '{modis_proj}'",
            " ".join(gcp_opts),
            f'HDF5:"{mosaic_file}"://fpar {mosaic_vrt_file}',
        )
    )

    logger.debug(f"{date} gdal_translate cmd: {cmd}")

    check_output(shlex.split(cmd))

    execute_gdalwarp = True
    if regridded_file.is_file():
        if recalculate or overwrite:
            logger.info(f"'{regridded_file}' exists. Deleting.")
            regridded_file.unlink()
        else:
            logger.warning(
                f"'{regridded_file}' exists and '{mosaic_file}' was not changed. "
                "Not executing gdalwarp."
            )
            execute_gdalwarp = False

    if execute_gdalwarp:
        cmd = " ".join(
            (
                f"{gdalwarp} -s_srs '{modis_proj}' -t_srs EPSG:4326 -ot Float32",
                "-srcnodata 255 -dstnodata -1",
                "-r average",
                *(("-multi",) if multi else ()),
                "-te -180 -90 180 90 -ts 1440 720",
                f"-wm {memory}",
                f"-of netCDF {mosaic_vrt_file} {regridded_file}",
            )
        )
        logger.debug(f"{date} gdalwarp cmd: {cmd}")
        check_output(shlex.split(cmd))

    if output_file.is_file():
        if execute_gdalwarp or overwite:
            logger.info(f"'{output_file}' exists. Deleting.")
            output_file.unlink()
        else:
            logger.warning(
                f"'{output_file}' exists and '{regridded_file}' was not changed. "
                "Not carrying out final processing."
            )
            return output_file

    # Read the regridded file, apply scaling factor, change metadata, and write to the
    # output file.
    cube = iris.load_cube(str(regridded_file))
    cube *= 0.01
    cube.var_name = None
    cube.standard_name = None
    cube.long_name = "Fraction of Absorbed Photosynthetically Active Radiation"
    cube.units = "1"
    iris.save(cube, output_file, zlib=False)

    logger.info(f"Finished writing to '{output_file}'.")
    return output_file


def get_file_dates(data_dir, start_date=None, end_date=None):
    """Get filenames and date strings for the given directory.

    Args:
        data_dir (str or pathlib.Path): Data directory.
        start_date (str, int, or None): Start date (inclusive).
        end_date (str, int, or None): End date (exclusive).

    Returns:
        dict: Ordered (ascending dates) mapping from date to file lists containing
            the corresponding data.

    Raises:
        ValueError: If either `start_date` or `end_date` is not None and not formatted
            like e.g. '2021034'.

    """
    if start_date is None:
        start_date = 1000000
    else:
        start_date = int(start_date)
    if end_date is None:
        end_date = 9999999
    else:
        end_date = int(end_date)

    if len(str(start_date)) != 7 or len(str(end_date)) != 7:
        raise ValueError(
            "Dates need to be formatted like '2012034'. "
            f"Got '{start_date}' and '{end_date}'."
        )

    files = list(Path(data_dir).glob(f"MOD15A2H.A*.hdf"))
    dates = [f.stem.split(".")[1][1:] for f in files]
    file_map = defaultdict(list)
    for f, date in zip(files, dates):
        if int(date) < start_date or int(date) >= end_date:
            continue
        file_map[date].append(f)
    sorted_dates = sorted(file_map, key=int)
    return {date: file_map[date] for date in sorted_dates}


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--memory", type=int, help="GDAL memory (in MB)", default=1000)
    parser.add_argument(
        "--start-date",
        help="start date (inclusive), e.g. 2012032",
        default=None,
    )
    parser.add_argument(
        "--end-date",
        help="end date (exclusive), e.g. 2021350",
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="source data directory",
        default=Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006",
    )
    parser.add_argument(
        "--temporary-dir",
        type=Path,
        help="directory for temporary files",
        default=Path(os.environ["EPHEMERAL"]) / "reprojection_tmp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory",
        default=Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006_0d25",
    )
    parser.add_argument(
        "--multi", action="store_true", help="add '-multi' to gdalwarp command"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing files"
    )
    parser.add_argument(
        "--gdal_translate", help="gdal_translate command path", default="gdal_translate"
    )
    parser.add_argument("--gdalwarp", help="gdalwarp command path", default="gdalwarp")

    args = parser.parse_args()

    enable_logging(level="DEBUG")

    args.temporary_dir.mkdir(exist_ok=True)
    args.output_dir.mkdir(exist_ok=True)

    file_map = get_file_dates(args.data_dir, args.start_date, args.end_date)

    for date, date_files in tqdm(file_map.items(), desc="Processing MODIS fPAR"):
        mosaic_process_date(
            date,
            date_files,
            args.temporary_dir,
            args.output_dir,
            args.memory,
            args.multi,
            args.overwrite,
            args.gdal_translate,
            args.gdalwarp,
        )
