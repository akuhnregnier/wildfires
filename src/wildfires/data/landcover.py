# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path

import dask.array as da
import iris
import numpy as np
import yaml
from numba import njit

# Load PFT conversion table as in Forkel et al. (2017).
with (Path(__file__).parent / "conversion_table.yaml").open("r") as f:
    conversion = yaml.safe_load(f)


def get_mapping_pfts(mapping):
    """Get all PFT names from the mapping."""
    pft_names = set()
    for value in mapping.values():
        pft_names.update(value["pfts"])
    return sorted(pft_names)


def get_mapping_arrays(pft_names, mapping):
    """Convert all PFT fractions into arrays in the given order."""
    converted = deepcopy(mapping)
    for value in converted.values():
        value["pfts"] = np.array(
            [value["pfts"].get(pft_name, 0) for pft_name in pft_names],
            dtype=np.uint8,
        )
    return converted


def convert_to_pfts(category_cube, conversion, min_category, max_category):
    """Convert landcover categories to PFT fractions using a given conversion table.

    Args:
        category_cube (iris.cube.Cube): Cube containing the landcover categories.
        conversion (dict): Conversion factors from categories to PFT fractions.
        min_category (int): Minimum possible land cover category index (inclusive).
        max_category (int): Maximum possible land cover category index (inclusive).

    Returns:
        iris.cube.CubeList: Cubes containing the PFTs on the same grid as
            `category_cube`.

    """
    if not category_cube.has_lazy_data():
        raise ValueError("Source cube needs to have lazy data.")

    pft_names = get_mapping_pfts(conversion)
    array_mapping = get_mapping_arrays(pft_names, conversion)

    n_pfts = next(iter(array_mapping.values()))["pfts"].size
    if not all(values["pfts"].size == n_pfts for values in array_mapping.values()):
        raise ValueError(
            "All categories need to map on to the same number of PFT fractions."
        )

    # Simple array structure containing the mapping from landcover categories to PFTs in a
    # way that is easier to accelerate.
    structured_mapping = np.zeros(
        (max_category - min_category + 1, n_pfts), dtype=np.uint8
    )
    for landcover_index in range(min_category, max_category + 1):
        if landcover_index in array_mapping:
            structured_mapping[landcover_index] = array_mapping[landcover_index]["pfts"]
        else:
            structured_mapping[landcover_index] = np.zeros(n_pfts, dtype=np.uint8)

    @njit
    def _execute_mapping(category, structured_mapping, n_pfts):
        """Carry out conversion to PFT fractions."""
        pfts = np.zeros((*category.shape, *(n_pfts,)))
        for index in np.ndindex(category.shape):
            pfts[index] = structured_mapping[category[index]]
        return pfts

    pft_data = da.map_blocks(
        _execute_mapping,
        category_cube.core_data(),
        structured_mapping=structured_mapping,
        n_pfts=n_pfts,
        meta=np.array([], dtype=np.uint8),
        # We are only adding a dimension with size `n_pfts`. All other chunks remain.
        chunks=(*category_cube.core_data().chunks, (n_pfts,)),
        new_axis=category_cube.ndim,
        dtype=np.uint8,
    )

    cubes = iris.cube.CubeList()
    for i, pft_name in enumerate(pft_names):
        pft_cube = category_cube.copy(data=pft_data[..., i])
        pft_cube.var_name = None
        pft_cube.standard_name = None
        pft_cube.long_name = pft_name
        pft_cube.units = "1"
        cubes.append(pft_cube)
    return cubes
