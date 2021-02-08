# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path

import dask.array as da
import iris
import numpy as np
import yaml

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
            dtype=np.float64,
        )
    return converted


def convert_to_pfts(category_cube, pft_names, conversion):
    """Convert landcover categories to PFT fractions using a given conversion table.

    Args:
        category_cube (iris.cube.Cube): Cube containing the landcover categories.
        conversion (dict): Conversion factors from categories to PFT fractions.

    Returns:
        iris.cube.CubeList: Cubes containing the PFTs on the same grid as
            `category_cube`.

    """
    if not category_cube.has_lazy_data():
        raise ValueError("Source cube needs to have lazy data.")
    if not all(
        isinstance(values["pfts"], np.ndarray) for values in conversion.values()
    ):
        raise ValueError(
            "PFT fractions in the conversion mapping need to use numpy arrays."
        )
    n_pfts = next(iter(conversion.values()))["pfts"].size
    if not all(values["pfts"].size == n_pfts for values in conversion.values()):
        raise ValueError(
            "All categories need to map on to the same number of PFT fractions."
        )

    def _execute_mapping(category):
        """Carry out conversion to PFT fractions at a single point."""
        if category in conversion:
            return conversion[category]["pfts"]
        return np.zeros(n_pfts, dtype=np.float64)

    # Vectorize so the function can be applied to all points within an array with a
    # simple function call.
    _execute_mapping = np.vectorize(_execute_mapping, signature="()->(n)")

    pft_data = da.map_blocks(
        _execute_mapping,
        category_cube.core_data(),
        meta=np.array([]),
        chunks=(*(-1,) * category_cube.ndim, *(n_pfts,)),
        new_axis=category_cube.ndim,
        dtype=np.float64,
    )
    # NOTE: This is required to permit indexing below, but would compute() work here
    # already (this would save repeat calculation)?
    pft_data.compute_chunk_sizes()
    cubes = iris.cube.CubeList()
    for i, pft_name in enumerate(pft_names):
        pft_cube = category_cube.copy(data=pft_data[..., i])
        pft_cube.var_name = None
        pft_cube.standard_name = None
        pft_cube.long_name = pft_name
        pft_cube.units = "1"
        cubes.append(pft_cube)
    return cubes
