#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Using MERIS observed area data to constrain observations.

Several burned area datasets are investigated.

"""
import warnings
from copy import deepcopy

from joblib import Memory
from tqdm import tqdm

from wildfires.analysis.plotting import *
from wildfires.data.cube_aggregation import *
from wildfires.data.datasets import *
from wildfires.logging_config import enable_logging
from wildfires.utils import land_mask as get_land_mask

memory = Memory(DATA_DIR)


@memory.cache
def get_mean_burned_area(thresholds):
    ba_var_names = ["CCI MODIS BA", "GFED4 BA", "GFED4s BA", "MCD64CMQ BA"]
    meris_var_names = ["CCI MERIS BA", "fraction of observed area"]
    ba_datasets = Datasets(
        (CCI_BurnedArea_MODIS_5_1(), GFEDv4(), GFEDv4s(), MCD64CMQ_C6())
    ).select_variables(ba_var_names) + Datasets(
        CCI_BurnedArea_MERIS_4_1()
    ).select_variables(
        meris_var_names
    )

    monthly = prepare_selection(ba_datasets, which="monthly")
    meris_obs_dataset = monthly.select_variables(
        meris_var_names[1], inplace=False
    ).dataset
    monthly = monthly.select_variables(ba_var_names + meris_var_names[0:1]).apply_masks(
        ~get_land_mask()
    )

    mean_bas = dict(
        (dataset_name, []) for dataset_name in ba_datasets.pretty_dataset_names
    )
    naive_mean_bas = deepcopy(mean_bas)
    valid_counts = deepcopy(mean_bas)

    for thres in tqdm(thresholds, desc="Thresholds", unit="threshold"):
        # For every threshold, first create the mask and apply it to each dataset.
        # All data has been converted to monthly data above and spans the same months.
        obs_mask = meris_obs_dataset.get_observed_mask(thres=thres, frequency="monthly")

        # Create a fresh copy to apply each mask onto.
        masked_datasets = monthly.copy(deep=True)

        # Apply the observed area mask.
        masked_datasets.apply_masks(obs_mask.data)

        # Modify the weights to take into account the observed area fraction.
        naive_weights = iris.analysis.cartography.area_weights(meris_obs_dataset.cube)
        weights = naive_weights * meris_obs_dataset.cube.data.data

        for ba_dataset, dataset_name in zip(
            tqdm(masked_datasets, desc="Datasets", unit="dataset"),
            masked_datasets.pretty_dataset_names,
        ):
            # Count the number of valid observations.
            valid_counts[dataset_name].append(np.sum(~ba_dataset.cube.data.mask))

            # Calculate the mean burned area.
            mean_bas[dataset_name].append(
                ba_dataset.cube.collapsed(
                    ("time", "latitude", "longitude"),
                    iris.analysis.MEAN,
                    weights=weights,
                ).data
            )

            # Comparison with the naive calculation.
            naive_mean_bas[dataset_name].append(
                ba_dataset.cube.collapsed(
                    ("time", "latitude", "longitude"),
                    iris.analysis.MEAN,
                    weights=naive_weights,
                ).data
            )

    return valid_counts, mean_bas, naive_mean_bas


if __name__ == "__main__":
    enable_logging()

    FigureSaver.directory = os.path.expanduser(
        os.path.join("~", "tmp", "observed_area")
    )
    os.makedirs(FigureSaver.directory, exist_ok=True)
    FigureSaver.debug = True

    warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
    warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")

    thresholds = np.round(np.linspace(0.1, 0.96, 15), 3)

    valid_counts, mean_bas, naive_mean_bas = get_mean_burned_area(thresholds)

    means_fits = {}
    naive_means_fits = {}

    column_names = ["gradient", "intercept", "gradient uncert.", "intercept uncert."]

    for bas, fit_dict in zip(
        (mean_bas, naive_mean_bas), (means_fits, naive_means_fits)
    ):

        for dataset_name, mean_ba_vals in bas.items():
            fit_dict[dataset_name] = pd.Series()

            p, V = np.polyfit(thresholds, mean_ba_vals, 1, cov=True)
            stds = np.sqrt(np.diag(V))

            for name, value in zip(column_names, list(p) + list(stds)):
                fit_dict[dataset_name][name] = value

    re_column_names = ["gradient", "gradient uncert.", "intercept", "intercept uncert."]

    means_fits = (
        pd.DataFrame(means_fits)
        .T[re_column_names]
        .sort_values("intercept", ascending=False)
    )
    naive_means_fits = (
        pd.DataFrame(naive_means_fits).T[re_column_names].loc[means_fits.index]
    )

    for df, fname in zip(
        (means_fits, naive_means_fits), ("means_fits.csv", "naive_means_fits.csv")
    ):
        df.to_csv(os.path.join(FigureSaver.directory, fname))

    with FigureSaver("mean_ba"):
        plt.figure()
        for i, dataset_name in enumerate(means_fits.index):
            mean_ba_vals = naive_mean_bas[dataset_name]
            plt.plot(thresholds, mean_ba_vals, linestyle="--", c=f"C{i}")
        for i, dataset_name in enumerate(means_fits.index):
            mean_ba_vals = mean_bas[dataset_name]
            plt.plot(thresholds, mean_ba_vals, label=dataset_name, c=f"C{i}")

        plt.annotate(
            "naive",
            xy=(0.48, 0.0018),
            xytext=(0.6, 0.00155),
            arrowprops=dict(arrowstyle="simple"),
            bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.15"),
        )
        plt.legend(loc="lower left")
        plt.xlabel("Threshold (1)")
        plt.ylabel("Mean Burned Area (1)")
        plt.show()

    with FigureSaver("counts"):
        plt.figure()
        for i, dataset_name in enumerate(means_fits.index):
            valid_count_vals = valid_counts[dataset_name]
            plt.plot(thresholds, valid_count_vals, label=dataset_name)
        plt.legend(loc="best")
        plt.xlabel("Threshold (1)")
        plt.ylabel("Valid Observations (1)")
        plt.show()
