#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import matplotlib as mpl
import numpy as np
from joblib import Memory
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from ..data import *
from ..joblib.cloudpickle_backend import register_backend
from ..logging_config import enable_logging
from ..qstat import get_ncpus
from .analysis import *
from .cx1_fitting import CX1Fit
from .plotting import *

__all__ = ("rf_time_lag_grid_search",)

logger = logging.getLogger(__name__)

register_backend()
location = os.path.join(DATA_DIR, "joblib_cachedir")
memory = Memory(location, backend="cloudpickle", verbose=100)

# Creating the Data Structures used for Fitting


@memory.cache
def get_data(shift_months=None, selection_variables=None):
    target_variable = "GFED4 BA"

    # Creation of new variables.
    transformations = {
        "Temp Range": lambda exog_data: (exog_data["Max Temp"] - exog_data["Min Temp"])
    }
    # Variables to be deleted after the aforementioned transformations.
    deletions = ("Min Temp",)

    # Variables required for the above.
    required_variables = ["Max Temp", "Min Temp", target_variable]

    # Carry out transformations, replacing old variables in the process.
    # log_var_names = ["Temp Range", "Dry Day Period"]
    # sqrt_var_names = [
    #     # "Lightning Climatology",
    #     "popd"
    # ]

    # Dataset selection.

    # TODO: Make this selection process more elegant.

    selection_datasets = [
        AvitabileThurnerAGB(),
        CHELSA(),
        Copernicus_SWI(),
        ERA5_CAPEPrecip(),
        ERA5_DryDayPeriod(),
        ESA_CCI_Landcover_PFT(),
        GFEDv4(),
        GlobFluo_SIF(),
        HYDE(),
        MOD15A2H_LAI_fPAR(),
        VODCA(),
    ]
    if shift_months is not None:
        datasets_to_shift = (ERA5_DryDayPeriod, MOD15A2H_LAI_fPAR, VODCA)
        for shift in shift_months:
            for shift_dataset in datasets_to_shift:
                selection_datasets.append(
                    shift_dataset.get_temporally_shifted_dataset(months=-shift)
                )

    if selection_variables is None:
        selection_variables = [
            "AGB Tree",
            "Max Temp",
            "Min Temp",
            "SWI(1)",
            "CAPE x Precip",
            "Dry Day Period",
            "ShrubAll",
            "TreeAll",
            "pftCrop",
            "pftHerb",
            "SIF",
            "popd",
            "FAPAR",
            "LAI",
            "VOD Ku-band",
        ]
        if shift_months is not None:
            for shift in shift_months:
                selection_variables.extend(
                    [
                        f"LAI {-shift} Month",
                        f"FAPAR {-shift} Month",
                        f"Dry Day Period {-shift} Month",
                        f"VOD Ku-band {-shift} Month",
                    ]
                )

    selection_variables = list(set(selection_variables).union(required_variables))

    selection = Datasets(selection_datasets).select_variables(selection_variables)
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which="climatology",
        transformations=transformations,
        deletions=deletions,
        # log_var_names=log_var_names,
        # sqrt_var_names=sqrt_var_names,
        use_lat_mask=False,
        use_fire_mask=False,
        target_variable=target_variable,
    )
    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


def print_header(ncol=70, char="#", fill=""):
    print(char * ncol)
    name_str = np.array(list("#" * ncol))
    n_fill = len(fill)
    start_index = int((ncol / 2) - (n_fill / 2)) - 1
    end_index = start_index + n_fill + 2
    name_str[start_index:end_index] = list(f" {fill} ")
    print("".join(name_str))
    print(char * ncol)


def rf_time_lag_grid_search():
    # Hyperparameter Optimisation Using CX1
    for shift_months, data_name in zip(
        (None, [1, 3, 6, 12, 24]), ("full_no_shift", "full_shift")
    ):
        logger.info(f"RF with data: {data_name}.")

        print_header(fill=data_name)

        (
            endog_data,
            exog_data,
            master_mask,
            filled_datasets,
            masked_datasets,
            land_mask,
        ) = get_data(shift_months=shift_months)

        # Define the training and test data.
        X_train, X_test, y_train, y_test = train_test_split(
            exog_data, endog_data, random_state=1, shuffle=True, test_size=0.3
        )

        # Define the parameter space.
        parameters_RF = {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [3, 10, 20],
            "max_features": ["auto"],
            "bootstrap": [False, True],
            "random_state": [1],
        }

        fitting = CX1Fit(
            X_train, y_train, data_name=data_name, param_grid=parameters_RF
        )
        fitting.run_job()
        output = fitting.get_best_model(timeout=60 * 60)
        if output:
            logger.info("Output found")
            regr = output["model"]

            regr.n_jobs = get_ncpus()
            print("RF n_jobs:", regr.n_jobs)

            print(regr)
            regr.predict(X_test)

            # Carry out predictions on the training dataset to diagnose overfitting.
            regr.predict(X_train)

            results = {}
            results["R2_train"] = regr.score(X_train, y_train)
            results["R2_test"] = regr.score(X_test, y_test)

            model_name = "RF"
            print(f"{model_name} R2 train: {results['R2_train']}")
            print(f"{model_name} R2 test: {results['R2_test']}")

            importances = regr.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in regr.estimators_], axis=0
            )

            importances_df = pd.DataFrame(
                {
                    "Name": exog_data.columns.values,
                    "Importance": importances,
                    "Importance STD": std,
                    "Ratio": np.array(std) / np.array(importances),
                }
            )
            print(
                "\n"
                + str(
                    importances_df.sort_values("Importance", ascending=False).to_string(
                        index=False, float_format="{:0.3f}".format, line_width=200
                    )
                )
            )

            print("VIFs")
            print_vifs(exog_data)

            with FigureSaver(
                [f"pdp_{data_name}_{feature}" for feature in X_test.columns]
            ):
                fig_axes = partial_dependence_plot(
                    regr,
                    X_test,
                    X_test.columns,
                    n_cols=4,
                    grid_resolution=70,
                    coverage=0.05,
                    predicted_name="burned area",
                    single_plots=True,
                )
                plt.subplots_adjust(wspace=0.16)

        else:
            logger.info("No output found")


if __name__ == "__main__":
    enable_logging()

    warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
    warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
    warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

    normal_coast_linewidth = 0.5
    mpl.rc("figure", figsize=(14, 6))
    mpl.rc("font", size=9.0)

    np.random.seed(1)

    shift_months = [1, 3, 6, 12, 24]

    selection_variables = (
        "VOD Ku-band -3 Month",
        "SIF",
        "VOD Ku-band -1 Month",
        "Dry Day Period -3 Month",
        "FAPAR",
        "pftHerb",
        "LAI -1 Month",
        "popd",
        "Dry Day Period -24 Month",
        "pftCrop",
        "FAPAR -1 Month",
        "FAPAR -24 Month",
        "Max Temp",
        "Dry Day Period -6 Month",
        "VOD Ku-band -6 Month",
        # Extra 5 split.
        # "Dry Day Period -1 Month",
        # "FAPAR -6 Month",
        # "ShrubAll",
        # "SWI(1)",
        # "TreeAll",
    )

    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = get_data(shift_months=shift_months, selection_variables=selection_variables)

    n_vars = len(exog_data.columns)

    data_name = f"clim_{n_vars}{'_shifted' if shift_months is not None else ''}"

    FigureSaver.directory = os.path.expanduser(
        os.path.join("~", "tmp", "time_lags", data_name)
    )
    os.makedirs(FigureSaver.directory, exist_ok=True)
    FigureSaver.debug = True

    # Define the training and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        exog_data, endog_data, random_state=1, shuffle=True, test_size=0.3
    )

    # Define the parameter space.
    parameters_RF = {
        "n_estimators": 100,
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

    y_pred = regr.predict(X_test)

    # Carry out predictions on the training dataset to diagnose overfitting.
    y_pred_train = regr.predict(X_train)

    print(regr)

    results = {}
    results["R2_train"] = regr.score(X_train, y_train)
    results["R2_test"] = regr.score(X_test, y_test)

    model_name = "RF"
    print(f"{model_name} R2 train: {results['R2_train']}")
    print(f"{model_name} R2 test: {results['R2_test']}")

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)

    importances_df = pd.DataFrame(
        {
            "Name": exog_data.columns.values,
            "Importance": importances,
            "Importance STD": std,
            "Ratio": np.array(std) / np.array(importances),
        }
    )
    print(
        "\n"
        + str(
            importances_df.sort_values("Importance", ascending=False).to_string(
                index=False, float_format="{:0.3f}".format, line_width=200
            )
        )
    )

    print("VIFs")
    print_vifs(exog_data)

    with FigureSaver([f"pdp_{data_name}_{feature}" for feature in X_test.columns]):
        fig_axes = partial_dependence_plot(
            regr,
            X_test,
            X_test.columns,
            n_cols=4,
            grid_resolution=50,
            coverage=0.6,
            predicted_name="burned area",
            single_plots=True,
            log_x_scale=("Dry Day Period", "popd"),
            X_train=X_train,
            plot_range=False,
        )
        plt.subplots_adjust(wspace=0.16)
