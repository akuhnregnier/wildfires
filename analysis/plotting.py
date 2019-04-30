#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import logging.config
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from wildfires.logging_config import LOGGING

logger = logging.getLogger(__name__)


def partial_dependence_plot(
    model,
    X,
    features,
    n_cols=3,
    grid_resolution=100,
    percentiles=(0.05, 0.95),
    coverage=1,
    random_state=None,
    predicted_name="burned area",
):
    """Plot 1 dimensional partial dependence plots.

    Args:
        model: A fitted model with a 'predict' method.
        X (pandas.DataFrame): Dataframe containing the data over which the
            partial dependence plots are generated.
        features (list): A list of int or string with which to select
            columns from X. The type must match the column labels.
        n_cols (int): Number of columns in the resulting plot.
        grid_resolution (int): Number of points to plot between the
            specified percentiles.
        percentiles (tuple of float): Percentiles between which to plot the
            pdp.
        coverage (float): A float between 0 and 1 which dictates what
            fraction of data is used to generate the plots.
        random_state (int or None): Used to call np.random.seed(). This is
            only used if coverage < 1. Supplying None causes the random
            number generator to initialise itself using the system clock
            (or using something similar which will not be constant from one
            run to the next).
        predicted_name (string): Name on the y axis.

    Returns:
        fig, ax: Figure and axes holding the pdp.

    """
    features = list(features)

    quantiles = X[features].quantile(percentiles)
    quantile_data = pd.DataFrame(
        np.linspace(quantiles.iloc[0], quantiles.iloc[1], grid_resolution),
        columns=features,
    )

    datasets = []

    if not np.isclose(coverage, 1):
        logger.debug("Selecting subset of data with coverage:{:}".format(coverage))
        np.random.seed(random_state)

        # Select a random subset of the data.
        permuted_indices = np.random.permutation(np.arange(X.shape[0]))
        permuted_indices = permuted_indices[: int(coverage * X.shape[0])]
    else:
        logger.debug("Selecting all data.")
        # Select all of the data.
        permuted_indices = np.arange(X.shape[0])

    X_selected = X.iloc[permuted_indices]

    for feature in tqdm(features):
        series = quantile_data[feature]

        # for each possible value of the selected feature, average over all
        # possible combinations of the other features
        averaged_predictions = []
        calc_X = X_selected.copy()
        for value in series:
            calc_X[feature] = value
            predictions = model.predict(calc_X)
            averaged_predictions.append(np.mean(predictions))

        datasets.append(np.array(averaged_predictions).reshape(-1, 1))

    datasets = np.hstack(datasets)
    results = pd.DataFrame(datasets, columns=features)

    fig, axes = plt.subplots(
        nrows=int(math.ceil(float(len(features)) / n_cols)), ncols=n_cols, squeeze=False
    )

    axes = axes.flatten()

    for (i, (ax, feature)) in enumerate(zip(axes, features)):
        ax.plot(quantile_data[feature], results[feature])
        ax.set_xlabel(feature)
        if i % n_cols == 0:
            ax.set_ylabel(predicted_name)

    for ax in axes[len(features) :]:
        ax.set_axis_off()

    plt.tight_layout()

    return fig, axes


if __name__ == "__main__":
    import string

    logging.config.dictConfig(LOGGING)

    m = 10000
    n = 20

    class A:
        def predict(self, X):
            X.iloc[:, 0] *= -1
            return np.sum(X, axis=1)

    model = A()
    X = pd.DataFrame(np.random.random((m, n)), columns=list(string.ascii_lowercase)[:n])
    features = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    fig, axes = partial_dependence_plot(model, X, features, grid_resolution=100)
