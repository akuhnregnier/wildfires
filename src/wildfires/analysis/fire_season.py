#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis of fire seasons.

For the purpose of these analyses, burned area at each location is relative to the
maximum burned are observed at that location. This allows the use of a single unitless
threshold throughout.

The start & end months of the fire seasons and their variation are analysed temporally
and spatially. A varying threshold is used to determine when fires are significant in
order to avoid small, anomalous fire events from influencing the final estimate.

A mask that allows selecting fire season months only may be calculated as well.

"""
import logging
import os
from copy import deepcopy

import matplotlib as mpl
import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.ndimage import label
from tqdm import tqdm

from ..data import *
from ..logging_config import enable_logging
from .plotting import *

__all__ = ("thres_fire_season_stats",)

logger = logging.getLogger(__name__)


location = os.path.join(DATA_DIR, "joblib_fire_season_cache")
memory = Memory(location)


def get_fire_season(
    ba_data,
    thres,
    climatology=True,
    quiet=True,
    return_mask=False,
    return_fraction=False,
):
    """Determine the fire season from burned area data.

    The mask is respected by returning masked arrays that contain the original mask.

    The fire seasons are organised into clusters - contiguous blocks of 'significant'
    burned area (see `thres`).

    Args:
        ba_data (numpy.ma.core.MaskedArray): Burned area data. The time-coordinate
            (first axis) should have a length that is an integer multiple of 12.
        thres (float): Threshold [0, 1]. Defines when normalised burned area (divided
            by maximum) is significant.
        climatology (bool): If True, treat the data as a climatology by allowing
            wrap-around of burned area clusters around the beginning and end of the
            time coordinate.
        quiet (bool): If True, suppress progress meter.
        return_mask (bool): If True, return a boolean numpy array representing the
            significant fire clusters.
        return_fraction (bool): If True, return an array containing the fraction of
            times above the threshold which are contained within the main cluster.

    Returns:
        indices, season_start, season_end, season_duration: Description of the fire
            season at each location given by `indices`.

    Examples:
        >>> import numpy as np
        >>> data_shape = (12, 4, 3)
        >>> data = np.ma.MaskedArray(np.zeros(data_shape), mask=np.zeros(data_shape))
        >>> data.mask[:, :, -1] = True
        >>> data[[0, -1], 0, 0] = 1
        >>> data[[-2, -1], 0, 1] = 1
        >>> data[[0, 1], 1, 0] = 1
        >>> data[0, 1, 1] = 1
        >>> data[:, 2, 0] = 1
        >>> data[-1, 2, 1] = 1
        >>> data[[0, 4, 5, 6], 3, 0] = 1
        >>> data[[0, 4, 5, 6, -1], 3, 1] = 1
        >>> out = get_fire_season(data, 0.5, return_mask=True)
        >>> for i, j in zip(*np.where(~out[0].mask)):
        ...     print(
        ...         (i, j), f"{out[0][i, j]:>2d} {out[1][i, j]:>2d} {out[2][i, j]:>2d}"
        ...     )
        (0, 0) 11  0  2
        (0, 1) 10 11  2
        (1, 0)  0  1  2
        (1, 1)  0  0  1
        (2, 0)  0 11 12
        (2, 1) 11 11  1
        (3, 0)  4  6  3
        (3, 1)  4  6  3
        >>> mask = np.zeros(data_shape, dtype=np.bool_)
        >>> mask[[0, -1], 0, 0] = 1
        >>> mask[[-2, -1], 0, 1] = 1
        >>> mask[[0, 1], 1, 0] = 1
        >>> mask[0, 1, 1] = 1
        >>> mask[:, 2, 0] = 1
        >>> mask[-1, 2, 1] = 1
        >>> mask[[4, 5, 6], 3, 0] = 1
        >>> mask[[4, 5, 6], 3, 1] = 1
        >>> np.all(mask == out[3])
        True

    """
    # Make sure the proper number of months are given.
    assert ba_data.shape[0] % 12 == 0, "Need an integer multiple of 12 months."

    # Make a copy of the mask initially, because certain operations may change this
    # later on.
    orig_mask = deepcopy(ba_data.mask)

    def null_func(x, *args, **kwargs):
        return x

    if return_mask:
        season_mask = np.zeros(ba_data.shape, dtype=np.bool_)

    # Normalise burned areas, dividing by the maximum burned area for each location.
    ba_data /= np.max(ba_data, axis=0)

    # Find significant samples.
    ba_data = ba_data > thres

    # Define the structure such that only elements touching in the time-axis are
    # counted as part of the same cluster.

    # TODO: Modify this to take into account spatial connectivity as well?
    # Eg. a cluster may be contained past points of no burning due to adjacent
    # pixels burning during the gaps.

    structure = np.zeros((3, 3, 3), dtype=np.int64)
    structure[:, 1, 1] = 1

    # Scipy `label` does not take the mask into account, so set masked elements to
    # boolean False in the input.
    ba_data[ba_data.mask] = False
    ba_data.mask = orig_mask

    # NOTE: Iterate like `range(1, n_clusters + 1)` for cluster indices.
    clusters, n_clusters = label(ba_data, structure=structure)

    # The data mask is used the determine where calculations should take place -
    # locations which are always masked are never considered.
    indices = np.where(np.any(~orig_mask, axis=0) & np.any(clusters, axis=0))

    starts = []
    ends = []
    sizes = []

    if return_fraction:
        fractions = []

    equal_cluster_errors = 0

    if climatology:
        # Iterate only over relevant areas.
        for xy in tqdm(zip(*indices), total=len(indices[0]), disable=quiet):
            cluster = clusters[(slice(None),) + tuple(xy)]
            assert np.any(cluster)
            size = 0
            main_cluster_index = None
            for cluster_index in set(np.unique(cluster)) - {0}:
                new_size = np.sum(cluster == cluster_index)
                if new_size > size:
                    size = new_size
                    main_cluster_index = cluster_index

            # To handle wrap-around, first determine where this is relevant - only
            # where there is a cluster both at the beginning and the end.

            # Also ignore the case where there is only one complete cluster since that
            # is not a wrap-around case.
            potential_wrap = False
            if np.logical_and(cluster[0], cluster[-1]) and not all(
                edge_index == main_cluster_index
                for edge_index in (cluster[0], cluster[-1])
            ):
                wrap_size = sum(
                    np.sum(cluster == cluster_index)
                    for cluster_index in (cluster[0], cluster[-1])
                )
                if wrap_size == size:
                    equal_cluster_errors += 1
                    logger.debug("Equal cluster sizes detected. Ignoring both.")
                    continue

                if wrap_size > size:
                    potential_wrap = True
                    size = wrap_size
                    cluster_selection = np.logical_or(
                        cluster == cluster[0], cluster == cluster[-1]
                    )
                    selected_indices = np.where(cluster_selection)[0]
                    # In the case of the wrap-around, stick to the convention that the
                    # 'last' index is the start and vice versa, to maintain a
                    # contiguous cluster across the wrap.

                    # The 'start' is the first occurrence of the final cluster.
                    start = np.where(cluster == cluster[-1])[0][0]
                    # The 'end' is the last occurrence of the initial cluster.
                    end = np.where(cluster == cluster[0])[0][-1]

            if not potential_wrap:
                # If we are this point, then wrapping is not significant.
                cluster_selection = cluster == main_cluster_index
                selected_indices = np.where(cluster_selection)[0]
                start = selected_indices[0]
                end = selected_indices[-1]

            starts.append(start)
            ends.append(end)
            sizes.append(size)

            if return_mask:
                season_mask[(slice(None),) + tuple(xy)] = cluster_selection

            if return_fraction:
                fractions.append(size / np.sum(cluster > 0))

        if equal_cluster_errors:
            logger.warning(
                f"{equal_cluster_errors} equal cluster size(s) detected and ignored."
            )
    else:
        raise NotImplementedError("Check back later.")

    start_arr = np.ma.MaskedArray(
        np.zeros(ba_data.shape[1:], dtype=np.int64), mask=True
    )
    end_arr = np.zeros_like(start_arr)
    size_arr = np.zeros_like(start_arr)

    valid_mask = np.any(season_mask, axis=0)

    start_arr[valid_mask] = starts
    end_arr[valid_mask] = ends
    size_arr[valid_mask] = sizes

    return_vals = [start_arr, end_arr, size_arr]

    if return_mask:
        return_vals.append(season_mask)

    if return_fraction:
        fract_arr = np.zeros_like(start_arr, dtype=np.float64)
        fract_arr[valid_mask] = fractions
        return_vals.append(fract_arr)

    return tuple(return_vals)


@memory.cache()
def get_burned_area_datasets(min_time=None, max_time=None, which="climatology"):
    fire_datasets = Datasets(
        (
            GFEDv4s(),
            GFEDv4(),
            CCI_BurnedArea_MODIS_5_1(),
            MCD64CMQ_C6(),
            CCI_BurnedArea_MERIS_4_1(),
        )
    ).select_variables(
        ["CCI MERIS BA", "CCI MODIS BA", "GFED4 BA", "GFED4s BA", "MCD64CMQ BA"]
    )
    climatology = prepare_selection(
        fire_datasets, min_time=min_time, max_time=max_time, which=which
    )

    for dataset in climatology:
        dataset.homogenise_masks()

    return climatology


@memory.cache()
def thres_fire_season_stats(thres, min_time=None, max_time=None, which="climatology"):
    if which != "climatology":
        raise NotImplementedError("Check back later.")
    datasets = get_burned_area_datasets()
    outputs = Parallel(verbose=20)(
        delayed(get_fire_season)(
            dataset.cube.data,
            thres,
            quiet=False,
            return_mask=True,
            return_fraction=True,
        )
        for dataset in datasets
    )
    return [[dataset.name] + list(output) for dataset, output in zip(datasets, outputs)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Investigate how the threshold affects the estimates.
    # The threshold is a fraction, relative to the maximum BA.
    enable_logging()
    plt.close("all")

    FigureSaver.debug = True
    FigureSaver.directory = os.path.join(os.path.expanduser("~"), "tmp", "fire_season")
    os.makedirs(FigureSaver.directory, exist_ok=True)

    outputs = []

    for thres in tqdm(np.round(np.geomspace(1e-4, 1e-1, 10), 5)):
        outputs.append(thres_fire_season_stats(thres))

        for dataset_outputs in outputs[-1]:
            name = dataset_outputs[0]
            starts = dataset_outputs[1]
            ends = dataset_outputs[2]
            sizes = dataset_outputs[3]

            for plot_type, data, cmap in zip(
                ("start (month)", "end (month)", "length (months)"),
                (starts, ends, sizes),
                (*("twilight",) * 2, "brewer_RdYlBu_11_r"),
            ):
                with FigureSaver(
                    f"{name}_thres_{str(thres).replace('.', '_')}_{plot_type}"
                ):
                    mpl.rc("figure", figsize=(7.4, 3.3))
                    cube_plotting(
                        data,
                        coastline_kwargs=dict(linewidth=0.5),
                        cmap=cmap,
                        label=plot_type,
                        title=name,
                        boundaries=np.arange(0, 12),
                    )
