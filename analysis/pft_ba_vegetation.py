#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import matplotlib as mpl
import numpy as np
from joblib import Memory
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from wildfires.analysis.analysis import *
from wildfires.analysis.plotting import *
from wildfires.analysis.time_lags import get_data
from wildfires.data.cube_aggregation import *
from wildfires.data.datasets import *
from wildfires.logging_config import enable_logging

logger = logging.getLogger(__name__)

location = os.path.join(DATA_DIR, "joblib_cachedir")
memory = Memory(location)


@memory.cache()
def get_fitted_model(shift_months=None, selection_variables=None):
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = get_data(shift_months=shift_months, selection_variables=selection_variables)

    # Define the training and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        exog_data, endog_data, random_state=1, shuffle=True, test_size=0.3
    )

    # Define the parameter space.
    parameters_RF = {
        "n_estimators": 10,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 3,
        "max_features": "auto",
        "bootstrap": True,
        "random_state": 1,
    }

    regr = RandomForestRegressor(**parameters_RF, n_jobs=get_ncpus())

    # Refit the model on all the data and store this as well.
    regr.fit(X_train, y_train)

    return regr, X_train, X_test, y_train, y_test, endog_data, exog_data, master_mask


if __name__ == "__main__":
    enable_logging()

    FigureSaver.directory = os.path.expanduser(
        os.path.join("~", "tmp", "pft_ba_vegetation")
    )
    os.makedirs(FigureSaver.directory, exist_ok=True)
    FigureSaver.debug = True

    warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
    warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
    warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

    normal_coast_linewidth = 0.5
    mpl.rc("figure", figsize=(14, 6))
    mpl.rc("font", size=9.0)

    np.random.seed(1)

    (
        regr,
        X_train,
        X_test,
        y_train,
        y_test,
        endog_data,
        exog_data,
        master_mask,
    ) = get_fitted_model()

    pfts = ESA_CCI_Landcover_PFT()
    pfts = pfts.get_climatology_dataset(start=pfts.min_time, end=pfts.max_time)

    # print('Mean Values')
    # for cube in pfts:
    #     cube.coord("latitude").guess_bounds()
    #     cube.coord("longitude").guess_bounds()

    #     print(
    #         format(
    #             cube.name(),
    #             '<12'
    #         ),
    #         ",",
    #         format(
    #             cube.collapsed(
    #                 ("time", "latitude", "longitude"),
    #                 iris.analysis.MEAN,
    #                 weights=iris.analysis.cartography.area_weights(cube),
    #             ).data,
    #             "0.4f",
    #         ),
    #     )

    pft_names = []
    stacked = []

    for name, cube in zip(pfts.variable_names("raw"), pfts):
        if name in ("pftBare", "pftNoLand"):
            continue
        pft_names.append(name)
        stacked.append(cube.data.reshape(1, -1))

    stacked = np.vstack(stacked)
    max_pfts = np.argmax(stacked, axis=0)

    print("Number of Dominances")
    for m in np.unique(max_pfts):
        print(pft_names[m], ",", np.sum(max_pfts == m))

    # masked = {}

    # for name, cube in zip(pfts.variable_names("raw"), pfts):
    #     if name in ('pftBare', 'pftNoLand'):
    #         continue

    #     if 'Tree' in name:
    #         key = 'pftTree'
    #         if key in masked:
    #             masked[key] += cube.data[master_mask]
    #         else:
    #             masked[key] = cube.data[master_mask]

    #     if 'Shrub' in name:
    #         key = 'pftShrub'
    #         if key in masked:
    #             masked[key] += cube.data[master_mask]
    #         else:
    #             masked[key] = cube.data[master_mask]

    # XXX: Or something like that.
    # for pft in pfts:
    #     data_name = "clim_{pft}"

    print("With master mask")
    for m in np.unique(max_pfts):
        pft_selection = max_pfts[~master_mask.reshape(-1)] == m

        # XXX:
        if pft_names[m] in (
            "pftCrop",
            "pftHerb",
            "pftShrubBD",
            "pftShrubBE",
            "ShrubAll",
            "TreeAll",
        ):
            continue

        print(pft_names[m], ",", np.sum(pft_selection))

        pft_dir = os.path.join(FigureSaver.directory, pft_names[m])
        os.makedirs(pft_dir, exist_ok=True)
        with FigureSaver(
            [f"pdp_{feature}" for feature in X_test.columns], directory=pft_dir
        ):
            fig_axes = partial_dependence_plot(
                regr,
                exog_data.loc[pft_selection],
                X_test.columns,
                grid_resolution=80,
                coverage=1.0,
                predicted_name="burned area",
                single_plots=True,
                log_x_scale=("Dry Day Period", "popd"),
                plot_range=False,
            )
            # XXX:
            # plt.subplots_adjust(wspace=0.16)

        plt.close("all")
