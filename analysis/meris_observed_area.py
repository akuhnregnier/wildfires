#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Using MERIS observed area data to constrain observations.

Several burned area datasets are investigated.

"""
import warnings
from copy import deepcopy

from tqdm import tqdm

from wildfires.analysis.plotting import *
from wildfires.data.cube_aggregation import *
from wildfires.data.datasets import *
from wildfires.logging_config import enable_logging
from wildfires.utils import land_mask as get_land_mask

if __name__ == "__main__":
    enable_logging()

    FigureSaver.directory = os.path.expanduser(os.path.join("~", "tmp", "to_send2"))
    os.makedirs(FigureSaver.directory, exist_ok=True)
    FigureSaver.debug = True

    warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
    warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")

    thresholds = np.round(np.linspace(0.1, 0.96, 15), 3)

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
    invalid_counts = deepcopy(mean_bas)

    for thres in tqdm(thresholds, desc="Thresholds", unit="threshold"):
        # For every threshold, first create the mask and apply it to each dataset.
        # All data has been converted to monthly data above and spans the same months.
        obs_mask = meris_obs_dataset.get_observed_mask(thres=thres, frequency="monthly")

        # Create a fresh copy to apply each mask onto.
        masked_datasets = monthly.copy(deep=True)

        # Apply the observed area mask.
        masked_datasets.apply_masks(obs_mask.data)

        for ba_dataset, dataset_name in zip(
            tqdm(masked_datasets, desc="Datasets", unit="dataset"),
            masked_datasets.pretty_dataset_names,
        ):
            # Count the number of valid observations.
            invalid_counts[dataset_name].append(np.sum(ba_dataset.cube.data.mask))

            # Calculate the mean burned area.
            mean_bas[dataset_name].append(
                ba_dataset.cube.collapsed(
                    ("time", "latitude", "longitude"),
                    iris.analysis.MEAN,
                    weights=iris.analysis.cartography.area_weights(ba_dataset.cube),
                ).data
            )

    with FigureSaver("mean_ba"):
        plt.figure()
        for dataset_name, mean_ba_vals in mean_bas.items():
            plt.plot(thresholds, mean_ba_vals, label=dataset_name)
        plt.legend(loc="best")
        plt.xlabel("Threshold (1)")
        plt.ylabel("Mean Burned Area (1)")
        plt.show()

    with FigureSaver("counts"):
        plt.figure()
        for dataset_name, valid_count_vals in invalid_counts.items():
            plt.plot(thresholds, valid_count_vals, label=dataset_name)
        plt.legend(loc="best")
        plt.xlabel("Threshold (1)")
        plt.ylabel("Invalid Observations (1)")
        plt.show()
