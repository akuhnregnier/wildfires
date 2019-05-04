#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions that aid the data analysis."""
import logging
import logging.config
from copy import deepcopy

import iris
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from sklearn.metrics import r2_score

from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


name_map = {
    "maximum temperature": "max temp",
    "Soil Water Index with T=1": "SWI",
    "VODorig": "VOD",
    "log temperature range": "log temp range",
    "log dry_day_period": "log dry period",
    "sqrt Combined Flash Rate Time Series": "sqrt Lightning",
}


log_set = {"burned area", "popd", "dry day period"}


def log_map(key, log_set=log_set):
    key = key.lower()
    if "log" in key:
        return False
    for search_str in (key, key.replace("_", " ")):
        if search_str in log_set or any(
            set_entry in search_str for set_entry in log_set
        ):
            return True
    return False


def log_modulus(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def map_name(name, name_map=name_map):
    return name_map.get(name, name)


def vif(exog_data):
    """Get a dataframe containing the VIFs for the input variables.

    Args:
        exog_data (pandas.DataFrame): One column per variable.

    Returns:
        pandas.DataFrame: Each row contains the variable name and its VIF.

    """
    vifs = []
    for i, name in enumerate(exog_data.columns):
        X_fit = exog_data.values[:, i].reshape(-1, 1)
        X_k = exog_data.values[:, [j for j in range(len(exog_data.columns)) if j != i]]
        X_k = np.hstack((np.ones(X_k.shape[0]).reshape(-1, 1), X_k))

        x, res, rank, s = np.linalg.lstsq(X_k, X_fit, rcond=None)
        predicted = X_k.dot(x)

        vif = 1.0 / (1 - r2_score(y_true=X_fit, y_pred=predicted))
        vifs.append(vif)

        # r2 = r2_score(y_true=X_fit, y_pred=predicted)
        # alt_r2 = OLS(X_fit, X_k).fit().rsquared
        # print('{:+>5.1e} {:+>5.1e}'.format(r2, alt_r2))
        # The r2 scores above do match, but the VIF values do not. This is likely due to
        # the fact that the statsmodels implementation uses the OLS linear fit procedure,
        # which does not add a constant by default (as was done explicitly above using
        # np.ones()). This is the only difference I could find, as the methods are otherwise
        # identical.

    return pd.DataFrame({"Name": exog_data.columns, "VIF": vifs})


def filling(cubes, land_mask, lat_mask):
    """Process cubes by filling gaps using NN interpolation and also filtering.

    Args:
        cubes: iris.cube.CubeList.
        land_mask
        lat_mask

    """
    # Check that all the cubes have masks
    assert np.all([hasattr(cube.data, "mask") for cube in cubes])
    # Respect the masking of 'monthly burned area' and ignore all others - for all others, replace
    # masked data using nearest-neighbour interpolation.
    # Thereafter, apply the land_mask and lat_mask, so that only data over land and within the latitude limits is considered.
    # Latitude limits due to anomalous behaviour of precipitation data, as well as limitations of the lightning LIS/OTD dataset.

    burned_area_cube = cubes.extract_strict(iris.Constraint(name="monthly burned area"))
    burned_area_mask = burned_area_cube.data.mask
    combined_mask = burned_area_mask | land_mask | lat_mask

    cubes_mod = deepcopy(cubes)
    assert isinstance(cubes_mod, iris.cube.CubeList)
    datasets_botched = 0
    for cube in cubes_mod:
        # In this part, data gaps are filled, so that the maximum possible area of data (limited by where burned area data is available)
        # is used for the analysis.
        # Choose to fill the gaps using nearest-neighbour interpolation.
        # To do this, define a mask which will tell the algorithm where to replace data.

        # Ignore burned area in this step, as this should never be modified!
        if cube.name() != "monthly burned area":
            print(cube.name())
            # Replace data where it is masked.
            fill_mask = cube.data.mask

            # Additional data replacing for datasets below.
            if cube.name() == "SIF":
                # Replace where it is above 20.
                fill_mask |= cube.data.data > 20
                # Replace where it is below 0.
                fill_mask |= cube.data.data < 0
                datasets_botched += 1
            elif cube.name() == "Combined Flash Rate Time Series":
                # Replace where it is below 0.
                fill_mask |= cube.data.data < 0
                datasets_botched += 1

            orig_data = cube.data.data.copy()
            if np.any(fill_mask):
                print(
                    "Filling {:} elements ({:} after final masking).".format(
                        np.sum(fill_mask), np.sum(fill_mask[~combined_mask])
                    )
                )
                filled_data = cube.data.data[
                    tuple(
                        nd.distance_transform_edt(
                            fill_mask, return_distances=False, return_indices=True
                        )
                    )
                ]
                assert np.all(
                    np.isclose(cube.data.data[~fill_mask], orig_data[~fill_mask])
                )

                selected_unfilled_data = orig_data[~combined_mask]
                selected_filled_data = filled_data[~combined_mask]

                print(
                    "Min {:0.1e}/{:0.1e}, max {:0.1e}/{:0.1e} before/after filling (for relevant regions)".format(
                        np.min(selected_unfilled_data),
                        np.min(selected_filled_data),
                        np.max(selected_unfilled_data),
                        np.max(selected_filled_data),
                    )
                )
            else:
                # Prevent overwriting with previous loop's filled data.
                filled_data = orig_data

            # Always apply global combined mask.
            cube.data = np.ma.MaskedArray(filled_data, mask=combined_mask)
            print("")
        else:
            # Also apply global combined mask to burned area.
            cube.data.mask = combined_mask

    # Assert that both SIF and Flash Rates were handled.
    assert datasets_botched == 2

    # Check that there aren't any inf's or nan's in the data.
    for cube in cubes_mod:
        assert not np.any(np.isinf(cube.data.data[~cube.data.mask]))
        assert not np.any(np.isnan(cube.data.data[~cube.data.mask]))

    return cubes_mod


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING)
