#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions that aid the data analysis."""
import logging
import logging.config

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

__all__ = ("log_map", "log_modulus", "map_name", "vif")

logger = logging.getLogger(__name__)


name_map = {
    "maximum temperature": "max temp",
    "Soil Water Index with T=1": "SWI",
    "VODorig": "VOD",
    "log temperature range": "log temp range",
    "log dry_day_period": "log dry period",
    "sqrt Combined Flash Rate Time Series": "sqrt Lightning",
    "sqrt Combined Flash Rate Monthly Climatology": "sqrt Lightning",
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


def vif(exog_data, verbose=False):
    """Get a dataframe containing the VIFs for the input variables.

    Args:
        exog_data (pandas.DataFrame): One column per variable.

    Returns:
        pandas.DataFrame: Each row contains the variable name and its VIF.

    """
    vifs = []
    for i, name in enumerate(
        tqdm(
            exog_data.columns,
            desc="Calculating VIFs",
            smoothing=0,
            disable=not verbose,
            unit="variable",
        )
    ):
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
