#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import logging.config
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
# import statsmodels.genmod.families.links as links

from wildfires.data.datasets import load_dataset_cubes
from wildfires.logging_config import LOGGING
logger = logging.getLogger(__name__)


def combine_masks(data, invalid_values=(0,)):
    """Create a mask that shows where data is invalid.

    NOTE: Calls data.data, so lazy data is realised here!

    True - invalid data
    False - valid data

    Returns a boolean array with the same shape as the input data.

    """
    if hasattr(data, 'mask'):
        mask = data.mask
        data_arr = data.data
    else:
        mask = np.zeros_like(data, dtype=bool)
        data_arr = data

    for invalid_value in invalid_values:
        mask |= np.isclose(data_arr, invalid_value)

    return mask


if __name__ == '__main__':
    logging.config.dictConfig(LOGGING)
    # TODO: Use iris cube long_name attribute to enter descriptive name
    # which will be used throughout data analysis and plotting (eg. for
    # selecting DataFrame columns).

    cubes = load_dataset_cubes()

    # TODO: Select desired cubes by name.

    # Use masking to extract only the relevant data.

    # Accumulate the masks for each dataset into a global mask.
    global_mask = np.zeros(cubes[0].shape, dtype=bool)
    for cube in cubes:
        # TODO: Find out the invalid values for the datasets that do not
        # have masks (if a mask hasn't been created before)!
        global_mask |= combine_masks(cube.data, invalid_values=[])

    # Apply the same mask for each latitude and longitude.
    collapsed_global_mask = np.any(global_mask, axis=0)
    global_mask = np.zeros_like(global_mask, dtype=bool)
    global_mask += collapsed_global_mask[np.newaxis]

    # Use this mask to select each dataset

    selected_datasets = []
    for cube in cubes:
        selected_data = cube.data[~global_mask]
        if hasattr(selected_data, 'mask'):
            assert not np.any(selected_data.mask)
            selected_datasets.append((cube.name(), selected_data.data))
        else:
            selected_datasets.append((cube.name(), selected_data))

    dataset_names = [s[0] for s in selected_datasets]

    exog_name_map = {
            'diurnal temperature range': 'diurnal temp range',
            'near-surface temperature minimum': 'near-surface temp min',
            'near-surface temperature': 'near-surface temp',
            'near-surface temperature maximum': 'near-surface temp max',
            'wet day frequency': 'wet day freq',
            'Volumetric Soil Moisture Monthly Mean': 'soil moisture',
            'SIF': 'SIF',
            'VODorig': 'VOD',
            'Combined Flash Rate Monthly Climatology': 'lightning rate',
            ('Population Density, v4.10 (2000, 2005, 2010, 2015, 2020)'
             ': 30 arc-minutes'): 'pop dens'
            }

    inclusion_names = {
            'near-surface temperature maximum',
            'Volumetric Soil Moisture Monthly Mean',
            'SIF',
            'VODorig',
            'diurnal temperature range',
            'wet day frequency',
            'Combined Flash Rate Monthly Climatology',
            ('Population Density, v4.10 (2000, 2005, 2010, 2015, 2020)'
             ': 30 arc-minutes'),
            }

    exog_names = [exog_name_map.get(s[0], s[0]) for s in
                  selected_datasets if s[0] in inclusion_names]
    raw_exog_data = np.hstack(
            [s[1].reshape(-1, 1) for s in selected_datasets
             if s[0] in inclusion_names])

    endog_name = selected_datasets[dataset_names.index('Burnt_Area')][0]
    endog_data = selected_datasets[dataset_names.index('Burnt_Area')][1]

    # lim = int(5e3)
    lim = None
    endog_data = endog_data[:lim]
    raw_exog_data = raw_exog_data[:lim]

    endog_data = pd.Series(endog_data, name='burned area')
    exog_data = pd.DataFrame(
            raw_exog_data,
            columns=exog_names)

    # TODO: Improve this by taking into account the number of days in each
    # month

    # Define dry days variable using the wet day variable.
    exog_data['dry day freq'] = 31.5 - exog_data['wet day freq']
    del exog_data['wet day freq']

    # Carry out log transformation for select variables.
    log_var_names = ['diurnal temp range',
                     # There are problems with negative surface
                     # temperatures here!
                     # 'near-surface temp max',
                     'dry day freq']

    for name in log_var_names:
        mod_data = exog_data[name] + 0.01
        assert np.all(mod_data > 0.01)
        exog_data['log ' + name] = np.log(mod_data)
        del exog_data[name]

    # Carry out square root transformation
    sqrt_var_names = ['lightning rate', 'pop dens']
    for name in sqrt_var_names:
        exog_data['sqrt ' + name] = np.sqrt(exog_data[name])
        del exog_data[name]

    '''
    # Available links for Gaussian:
    [statsmodels.genmod.families.links.log,
     statsmodels.genmod.families.links.identity,
     statsmodels.genmod.families.links.inverse_power]

    '''

    model = sm.GLM(endog_data, exog_data,
                   # family=sm.families.Gaussian(links.log)
                   family=sm.families.Binomial()
                   )

    model_results = model.fit()

    sm.graphics.plot_partregress_grid(model_results)
    plt.tight_layout()

    plt.figure()
    plt.hexbin(endog_data, model_results.fittedvalues, bins='log')
    plt.xlabel('real data')
    plt.ylabel('prediction')
