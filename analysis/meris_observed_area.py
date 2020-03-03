#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Using MERIS observed area data to constrain observations.

Several burned area datasets are investigated.

"""
import warnings
from copy import deepcopy

from joblib import Memory
from tqdm import tqdm

from wildfires.analysis.fire_season import thres_fire_season_stats
from wildfires.analysis.plotting import *
from wildfires.data.cube_aggregation import *
from wildfires.data.datasets import *
from wildfires.logging_config import enable_logging
from wildfires.utils import land_mask as get_land_mask

memory = Memory(DATA_DIR)


@memory.cache
def get_mean_burned_area(thresholds, masks=None, climatology_masks=None):
    """

    Climatology masks define a mask for each point on the grid for each of the twelve
    months. These individual-month masks then have to applied manually to the months
    in the monthly dataset.

    """
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

    monthly_masks = [~get_land_mask()]
    if masks is not None:
        monthly_masks.extend(masks)

    # Process climatological masks to match the original months to the target months.
    # NOTE: The masking process here relies on using `which='monthly'` above, which
    # should not be altered!
    start = monthly[0].min_time
    end = monthly[0].max_time

    n_repetitions = end.year - start.year + 1

    for mask in climatology_masks:
        repeated_mask = np.vstack((mask,) * n_repetitions)

        # Trim leftover months in case the first data point was not in January, and
        # the last was not in December.
        monthly_masks.append(
            ~repeated_mask[start.month - 1 : repeated_mask.shape[0] - (12 - end.month)]
        )
        assert monthly_masks[-1].shape == monthly[0][0].shape

    monthly = monthly.select_variables(ba_var_names + meris_var_names[0:1]).apply_masks(
        *monthly_masks
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

    # Normal case - only the land is ignored.
    masks = None
    climatology_masks = None

    # Use a climatological mask.
    season_output = thres_fire_season_stats(np.round(1e-3, 5))
    gfedv4_index = [s[0] for s in season_output].index("GFEDv4")
    climatology_masks = [season_output[gfedv4_index][4]]

    valid_counts, mean_bas, naive_mean_bas = get_mean_burned_area(
        thresholds, masks=masks, climatology_masks=climatology_masks
    )

    if masks is None and climatology_masks is None:
        fire_season = False
    else:
        fire_season = True

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

    prefix = "fire_season_" if fire_season else ""

    for df, fname in zip(
        (means_fits, naive_means_fits), ("means_fits.csv", "naive_means_fits.csv")
    ):
        df.to_csv(os.path.join(FigureSaver.directory, prefix + fname))

    mpl.rc("figure", figsize=(6.4, 4.8))

    with FigureSaver(prefix + "mean_ba"):
        plt.figure()
        for i, dataset_name in enumerate(means_fits.index):
            mean_ba_vals = naive_mean_bas[dataset_name]
            plt.plot(thresholds, mean_ba_vals, linestyle="--", c=f"C{i}")
        for i, dataset_name in enumerate(means_fits.index):
            mean_ba_vals = mean_bas[dataset_name]
            plt.plot(thresholds, mean_ba_vals, label=dataset_name, c=f"C{i}")

        plt.annotate(
            "naive",
            xy=(0.48, 0.0018) if not fire_season else (0.48, 0.017),
            xytext=(0.6, 0.00155) if not fire_season else (0.6, 0.0155),
            arrowprops=dict(arrowstyle="simple"),
            bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.15"),
        )
        plt.legend(loc="lower left")
        plt.xlabel("Threshold (1)")
        plt.ylabel("Mean Burned Area (1)")
        plt.show()

    with FigureSaver(prefix + "counts"):
        plt.figure()
        for i, dataset_name in enumerate(means_fits.index):
            valid_count_vals = valid_counts[dataset_name]
            plt.plot(thresholds, valid_count_vals, label=dataset_name)
        plt.legend(loc="best")
        plt.xlabel("Threshold (1)")
        plt.ylabel("Valid Observations (1)")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.show()
