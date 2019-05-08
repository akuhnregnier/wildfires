#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate aggregated cubes.

A subset of variables is selected. This selected subset, its total time average, and
its monthly climatology are all stored as pickle files.

"""
import logging
import logging.config
import os
import pickle

import fiona
import iris
import iris.coord_categorisation
import numpy as np
import rasterio
from affine import Affine
from rasterio import features
from tqdm import tqdm

from wildfires.data.datasets import DATA_DIR, load_dataset_cubes
from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)

target_pickles = tuple(
    os.path.join(DATA_DIR, filename)
    for filename in (
        "selected_cubes.pickle",
        "mean_cubes.pickle",
        "monthly_climatologies.pickle",
    )
)


def get_all_dataset_variables():
    import wildfires.data.datasets as datasets

    for name in dir(datasets):
        print(name)
        obj = getattr(datasets, name)
        if (
            obj != datasets.Dataset
            and hasattr(obj, "__mro__")
            and datasets.Dataset in obj.__mro__
        ):
            try:
                instance = obj()
                print([cube.name() for cube in instance.cubes])
            except NotImplementedError:
                print(name, "not implemented")


def aggregate_cubes():
    logger.info(
        "Checking for the existence of the target pickles: {}".format(target_pickles)
    )
    if all(os.path.isfile(filename) for filename in target_pickles):
        logger.info("All target pickles exist, not aggregating cubes.")
        return None

    logger.info("One or more target pickles did not exist, aggregating cubes.")
    logger.info("Loading cubes")
    cubes = load_dataset_cubes()

    # Get list of names for further selection.
    # from pprint import pprint
    # pprint([cube.name() for cube in cubes])

    selected_names = [
        "AGBtree",
        # 'mean temperature',
        # 'monthly precipitation',
        "maximum temperature",
        "minimum temperature",
        # 'Quality Flag with T=1',
        # 'Soil Water Index with T=60',
        # 'Soil Water Index with T=5',
        # 'Quality Flag with T=60',
        # 'Soil Water Index with T=20',
        # 'Soil Water Index with T=40',
        # 'Quality Flag with T=20',
        # 'Surface State Flag',
        "Soil Water Index with T=1",
        # 'Quality Flag with T=100',
        # 'Soil Water Index with T=15',
        # 'Quality Flag with T=10',
        # 'Soil Water Index with T=10',
        # 'Quality Flag with T=40',
        # 'Quality Flag with T=15',
        # 'Quality Flag with T=5',
        "Soil Water Index with T=100",
        "ShrubAll",
        "TreeAll",
        "pftBare",
        "pftCrop",
        "pftHerb",
        "pftNoLand",
        # 'pftShrubBD',
        # 'pftShrubBE',
        # 'pftShrubNE',
        # 'pftTreeBD',
        # 'pftTreeBE',
        # 'pftTreeND',
        # 'pftTreeNE',
        "monthly burned area",
        "dry_days",
        "dry_day_period",
        "precip",
        "SIF",
        # 'cropland',
        # 'ir_norice',
        # 'rf_norice',
        "popd",
        # 'conv_rangeland',
        # 'rangeland',
        # 'tot_rainfed',
        # 'pasture',
        # 'rurc',
        # 'rf_rice',
        # 'tot_rice',
        # 'uopp',
        # 'popc',
        # 'ir_rice',
        # 'urbc',
        # 'grazing',
        # 'tot_irri',
        "Combined Flash Rate Time Series",
        "VODorig",
        # 'Standard Deviation of LAI',
        "Fraction of Absorbed Photosynthetically Active Radiation",
        "Leaf Area Index",
        # 'Standard Deviation of fPAR',
        # 'Simard_Pinto_3DGlobalVeg_L3C',
        # 'biomass_totalag',
        # 'biomass_branches',
        # 'biomass_foliage',
        # 'biomass_roots',
        # 'biomass_stem'
    ]

    def selection_func(cube):
        return cube.name() in selected_names

    selected_cubes = cubes.extract(iris.Constraint(cube_func=selection_func))
    logger.info(selected_cubes)

    # Realise data - not necessary when using iris.save, but necessary here as pickle
    # files are being used!
    [c.data for c in selected_cubes]
    with open(target_pickles[0], "wb") as f:
        pickle.dump(selected_cubes, f, protocol=-1)

    mean_cubes = iris.cube.CubeList(
        [c.collapsed("time", iris.analysis.MEAN) for c in selected_cubes]
    )
    logger.info(mean_cubes)

    # Realise data - not necessary when using iris.save, but necessary here as pickle
    # files are being used!
    [c.data for c in mean_cubes]
    with open(target_pickles[1], "wb") as f:
        pickle.dump(mean_cubes, f, protocol=-1)

    # Generate monthly climatology.
    averaged_cubes = iris.cube.CubeList([])
    for cube in tqdm(selected_cubes):
        if not cube.coords("month_number"):
            iris.coord_categorisation.add_month_number(cube, "time")
        averaged_cubes.append(cube.aggregated_by("month_number", iris.analysis.MEAN))

    # Store monthly climatology.
    # Realise data - not necessary when using iris.save, but necessary here as pickle
    # files are being used!
    [c.data for c in averaged_cubes]
    with open(target_pickles[2], "wb") as f:
        pickle.dump(averaged_cubes, f, protocol=-1)

    return None


def land_mask(n_lon=1440):
    """Create land mask at the desired resolution.

    Data is taken from https://www.naturalearthdata.com/

    Args:
        n_lon (int): The number of longitude points of the final mask array. As the
            ratio between number of longitudes and latitudes has to be 2 in order for
            uniform scaling to work, the number of latitudes points is calculated as
            n_lon / 2.

    Returns:
        numpy.ndarray: Array of shape (n_lon / 2, n_lon) and dtype np.bool_. True
            where there is land, False otherwise.

    Examples:
        >>> import numpy as np
        >>> mask = land_mask(n_lon=1440)
        >>> mask.dtype == np.bool_
        True
        >>> mask.sum()
        343928
        >>> mask.shape
        (720, 1440)

    """
    assert n_lon % 2 == 0, (
        "The number of longitude points has to be an even number for the number of "
        "latitude points to be an integer."
    )
    n_lat = round(n_lon / 2)
    geom_np = np.zeros((n_lat, n_lon), dtype=np.uint8)
    with fiona.open(
        os.path.join(DATA_DIR, "land_mask", "ne_110m_land.shp"), "r"
    ) as shapefile:
        for geom in shapefile:
            geom_np += features.rasterize(
                [geom["geometry"]],
                out_shape=geom_np.shape,
                dtype=np.uint8,
                transform=~(
                    Affine.translation(n_lat, n_lat / 2) * Affine.scale(n_lon / 360)
                ),
            )

    geom_np = geom_np.astype(np.bool_)
    return geom_np


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
    # aggregate_cubes()
    get_all_dataset_variables()
