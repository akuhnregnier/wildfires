#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of code to be used throughout the project.

"""
import os

import fiona
import numpy as np
from affine import Affine
from rasterio import features

from wildfires.data.datasets import DATA_DIR


class RampVar:
    """Variable that is increased upon every call.

    The starting value, maximum value and the steps can be set.

    The value is incremented linearly between the initial and maximum
    value, with `steps` intervals.

    Args:
        initial_value (float): Initial value.
        max_value (float): Maximum value the variable can take.
        steps (int): The number of intervals.

    Examples:
        >>> var = RampVar(0, 2, 3)
        >>> int(round(var.value))
        0
        >>> int(round(var.value))
        1
        >>> int(round(var.value))
        2
        >>> var.reset()
        >>> int(round(var.value))
        0

    """

    def __init__(self, initial_value, max_value, steps=10):
        self.steps = steps
        self.values = np.linspace(initial_value, max_value, steps)
        self.index = -1

    @property
    def value(self):
        """Every time this attribute is accessed it is incremented as
        defined by the values given to the constructor.

        """
        if self.index < self.steps - 1:
            self.index += 1
        return self.values[self.index]

    def reset(self):
        """Resets the value to the initial value."""
        self.index = -1


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
