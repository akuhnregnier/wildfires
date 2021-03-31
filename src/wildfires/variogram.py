# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
from distributed import Client, as_completed
from sklearn.metrics.pairwise import haversine_distances
from tqdm.auto import tqdm

__all__ = (
    "binned_variance_batch",
    "combine_multiple_stats",
    "combine_stats",
    "compute_variogram",
    "plot_variogram",
)


def combine_stats(old_n, new_n, old_mean, new_mean, old_v, new_v):
    """Combine two mean and variance measurements.

    Args:
        old_n, new_n (int): Old and new number of samples.
        old_mean, new_mean (float): Old and new mean.
        old_v, new_v (float): Old and new variance.

    Returns:
        total_n (int): Combined number of samples.
        combined_mean (float): Combined mean.
        combined_v (float): Combined variance.

    """
    total_n = old_n + new_n
    combined_mean = (old_n / total_n) * old_mean + (new_n / total_n) * new_mean
    combined_v = (
        (old_n / total_n) * old_v
        + (new_n / total_n) * new_v
        + (old_n * new_n / (total_n ** 2)) * ((old_mean - new_mean) ** 2)
    )
    return total_n, combined_mean, combined_v


def combine_multiple_stats(ns, means, variances):
    """Combine multiple mean and variance measurements.

    Args:
        ns (iterable of int): Number of samples.
        means (iterable of float): Means.
        variances (iterable of float): Variances.

    Returns:
        n (int): Total number of samples.
        mean (float): Combined mean.
        variance (float): Combined variance.

    """
    combined_n = ns[0]
    combined_mean = means[0]
    combined_v = variances[0]
    for n, mean, variance in zip(ns[1:], means[1:], variances[1:]):
        combined_n, combined_mean, combined_v = combine_stats(
            combined_n, n, combined_mean, mean, combined_v, variance
        )
    return combined_n, combined_mean, combined_v


def binned_variance_batch(inds1, inds2, bin_edges, coords, X):
    # Compute distances.
    distances = haversine_distances(
        coords[inds1[0] : inds1[1]], coords[inds2[0] : inds2[1]]
    )

    # Convert distances to km.
    distances *= 6371000 / 1000

    # Use just the upper triangle later on - mark all others using -1.
    distances[
        np.triu_indices(
            n=distances.shape[0], m=distances.shape[1], k=abs(inds1[0] - inds2[0])
        )
    ] = -1

    n_samples = np.empty((bin_edges.shape[0] - 1,), dtype=np.int64)
    means = np.empty((bin_edges.shape[0] - 1,))
    variances = np.empty((bin_edges.shape[0] - 1,))

    for (bin_index, (lower, upper)) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # Bin the observations.
        selection = (lower <= distances) & (distances < upper)
        # Get matching indices.
        diffs = np.empty((np.sum(selection),))
        for (counter, (i, j)) in enumerate(zip(*np.where(selection))):
            diffs[counter] = X[inds1[0] + i] - X[inds2[0] + j]

        n = diffs.size
        n_samples[bin_index] = n
        means[bin_index] = np.mean(diffs) if n else 0
        variances[bin_index] = np.var(diffs) if n else 0

    return n_samples, means, variances


def compute_variogram(
    coords, X, bins=10, max_lag=150, n_jobs=1, n_per_job=10, verbose=False
):
    """Compute semivariances.

    Args:
        coords (array with shape (N, 2)): Data coordinates.
        X (array with shape (N,)): Data.
        bins (int): Number of bins in [0, `max_lag`].
        max_lag (float): Maximum distance (see above).
        n_jobs (int): Maximum concurrent number of jobs. Runtime is expected to scale
            approx. inversely with `n_jobs`, while memory usage is expected to scale
            approx. linearly with `n_jobs`. If `n_jobs=1`, chunks are processed in
            serial.
        n_per_job (int): Number of samples per job. Memory usage will be
            approx. `n_jobs * n_per_job ** 2`.
        verbose (bool): If True, show a progress bar.

    Returns:
        bin_edges (array of shape (`bins + 1`,)): Bin edges used for binning pairwise
            distances.
        semivariances (array of shape (`bins + 1`,)): Semivariances.

    """
    assert X.shape[0] == coords.shape[0]

    # Convert coordinates (lat, lon) to radians.
    coords = coords * np.pi / 180.0

    bin_edges = np.linspace(0, max_lag, bins + 1)

    n_batches = math.ceil(coords.shape[0] / n_per_job)

    batch_indices = np.unique(
        np.linspace(0, coords.shape[0], n_batches + 1, dtype=np.int64)
    )

    index_pairs = []

    for lower_index1, upper_index1 in zip(batch_indices[:-1], batch_indices[1:]):
        for lower_index2, upper_index2 in zip(batch_indices[:-1], batch_indices[1:]):
            # Discard those splits where the 'lower-left' vertex is on or above the
            # diagonal line (i==j) - we only want to compute elements for the lower
            # triangular pairwise distance matrix.
            if upper_index1 <= lower_index2:
                continue

            index_pairs.append(
                (
                    (lower_index1, upper_index1),
                    (lower_index2, upper_index2),
                )
            )

    batched_n_samples = []
    batched_means = []
    batched_variances = []

    if n_jobs == 1:
        for index_pair in tqdm(index_pairs, disable=not verbose):
            batch_ns, batch_means, batch_variances = binned_variance_batch(
                *index_pair,
                bin_edges,
                coords,
                X,
            )
            batched_n_samples.append(batch_ns)
            batched_means.append(batch_means)
            batched_variances.append(batch_variances)
    else:
        # Create a Dask LocalCluster.
        client = Client(n_workers=min(n_jobs, len(index_pairs)), threads_per_worker=1)

        # Scatter the shared data.
        scat_bin_edges = client.scatter(bin_edges)
        scat_coords = client.scatter(coords)
        scat_X = client.scatter(X)

        # Submit the jobs.
        futures = []
        for index_pair in index_pairs:
            futures.append(
                client.submit(
                    binned_variance_batch,
                    *index_pair,
                    scat_bin_edges,
                    scat_coords,
                    scat_X,
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures), disable=not verbose):
            batch_ns, batch_means, batch_variances = f.result()
            batched_n_samples.append(batch_ns)
            batched_means.append(batch_means)
            batched_variances.append(batch_variances)

        # Terminate the cluster.
        client.shutdown()
        client.close()

    combined_stats = []
    for ns, ms, vs in zip(
        zip(*batched_n_samples), zip(*batched_means), zip(*batched_variances)
    ):
        sel = np.asarray(ns) > 0
        if not np.any(sel):
            combined_stats.append((0, 0.0, 0.0))
            continue
        combined_stats.append(
            combine_multiple_stats(
                np.asarray(ns)[sel], np.asarray(ms)[sel], np.asarray(vs)[sel]
            )
        )
    new_samples, new_means, new_variances = map(np.asarray, zip(*combined_stats))
    return bin_edges, new_samples, 0.5 * new_variances


def plot_variogram(coords, X, **kwargs):
    bin_edges, counts, semivariances = compute_variogram(coords, X, **kwargs)

    fig = plt.figure()
    ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
    ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)

    ax1.plot(bin_edges, [0] + list(semivariances), marker="o", alpha=0.6)
    ax1.grid(alpha=0.4, linestyle="--")

    ax2.bar(
        bin_edges[1:],
        counts,
        width=0.75 * bin_edges[-1] / len(bin_edges),
        align="center",
        zorder=3,
    )
    ax2.grid(alpha=0.4, linestyle="--", zorder=0)
    plt.setp(ax2.axes.get_xticklabels(), visible=False)

    return fig, ax1, ax2
