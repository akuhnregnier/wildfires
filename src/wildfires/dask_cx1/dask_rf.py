# -*- coding: utf-8 -*-
import logging
import math
import pickle
import shutil
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import CancelledError
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from itertools import product
from numbers import Number
from operator import itemgetter
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Lock, Thread, Timer
from warnings import warn

import numpy as np
from dask.distributed import as_completed
from dask.utils import parse_timedelta
from joblib import parallel_backend
from sklearn import clone
from sklearn.ensemble._forest import (
    DOUBLE,
    DTYPE,
    MAX_INT,
    DataConversionWarning,
    RandomForestRegressor,
    _check_sample_weight,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
    check_random_state,
    issparse,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
from sklearn.model_selection._search import (
    GridSearchCV,
    _check_multimetric_scoring,
    _fit_and_score,
    check_cv,
    indexable,
    is_classifier,
)
from tqdm.auto import tqdm

from .dask_utils import common_worker_threads

__all__ = (
    "CachedResults",
    "DaskGridSearchCV",
    "DaskRandomForestRegressor",
    "dask_fit_combinations",
    "dask_fit_loco",
    "fit_dask_sub_est_grid_search_cv",
    "fit_dask_sub_est_random_search_cv",
    "temp_sklearn_params",
)

logger = logging.getLogger(__name__)


@contextmanager
def temp_sklearn_params(est, params):
    """Temporally alter a set of parameters.

    Args:
        est (object with `set_params()` and `get_params()` method): Scikit-learn
            estimator.
        params (dict): The parameters to temporarily alter.

    """
    original_params = est.get_params()
    try:
        est.set_params(**params)
        yield
    finally:
        # Reset the original parameters.
        est.set_params(**original_params)


class DaskRandomForestRegressor(RandomForestRegressor):
    def dask_fit(self, X, y, X_f, y_f, client, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Note that this implementation returns futures instead of `self`, which is most
        useful when fitting a series of trees in parallel, eg. using Dask, as is done
        in `fit_dask_sub_est_grid_search_cv` which `dask_fit()` was written for.

        For most use cases, using `fit()`, perhaps using the Dask parallel backend,
        would be preferable, as that could parallelise the operation over a cluster
        more easily and transparently:

            >>> from dask.distributed import Client  # doctest: +SKIP
            >>> client = Client(processes=False)  # doctest: +SKIP
            >>> from joblib import parallel_backend  # doctest: +SKIP
            >>> with parallel_backend("dask"):  # doctest: +SKIP
            ...     rf.fit(X, y)  # doctest: +SKIP

        Args:
            X (array-like of shape (n_samples, n_features)): The training input samples.
            y (array-like of shape (n_samples,) or (n_samples, n_outputs)): The target
                values.
            X_f (Future): Future pointing to `X` (see above).
            y_f (Future): Future pointing to `y` (see above).
            client (`dask.distributed.Client`): Dask Client used to submit the
                individual fit operations to the scheduler.
            sample_weight (array-like of shape (n_samples,)): Sample weights. If None,
                then samples are equally weighted. Splits that would create child
                nodes with net zero or negative weight are ignored while searching for
                a split in each node. In the case of classification, splits are also
                ignored if they would result in any single class carrying a negative
                weight in either child node.

        Returns:
            list of futures: Futures pointing to the trees being fit on the cluster as
                submitted by `client`.

        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError(
                "Out of bag estimation only available" " if bootstrap=True"
            )

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        tree_fs = []

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(self.n_estimators)
            ]

            # Do this using Dask now, return futures instead of the original trees.
            tree_fs = [
                client.submit(
                    _parallel_build_trees,
                    t,
                    self,
                    X_f,
                    y_f,
                    sample_weight,
                    i,
                    len(trees),
                    self.verbose,
                    self.class_weight,
                    n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            ]

            # NOTE: Since we are not waiting for the trees to be fit, this has to be
            # done later on!
            # Collect newly grown trees
            # self.estimators_.extend(trees)

        # NOTE: Since we are not waiting for the trees to be fit, this has to be
        # done later on!
        # if self.oob_score:
        #     self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return tree_fs


def fit_dask_sub_est_grid_search_cv(
    estimator,
    X,
    y,
    n_splits,
    param_grid,
    client,
    verbose=False,
    refit=True,
    return_train_score=True,
    local_n_jobs=None,
):
    """Carry out a grid search using Dask-adapted scikit-learn estimators.

    The futures returned by `dask_fit()` are tracked and if all sub-estimators
    belonging to an estimator have been trained successfully, the sub-estimators are
    collated and used to perform the scoring.

    Depending on the problem, this method may be slower than using `DaskGridSearchCV`.

    Args:
        estimator (object implementing `dask_fit()` and `score()` methods): Estimator
            to be evaluated.
        X (array-like): Training vector.
        y (array-like): Target relative to `X`.
        n_splits (int): Number of splits used for `KFold()`.
        param_grid (dict): Dictionary with parameter names (`str`) as keys and lists
            of parameter settings to try as values.
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.
        verbose (bool): If True, print out progress information related to the fitting
            of individual sub-estimators and scoring of the resulting estimators.
        refit (bool): If True, fit `estimator` using the best parameters on all of `X`
            and `y`.
        return_train_score (bool): If True, compute training scores.
        local_n_jobs (int): Since scoring has a 'sharedmem' requirement (ie. threading
            backend), parallelisation can be achieved locally using the threading
            backend with `local_n_jobs` threads.

    Returns:
        dict: Dictionary containing the test (and train) scores for individual
            parameters and splits.
        estimator: Only present if `refit` is True. `estimator` fit on `X` and `y`
            using the best parameters found.

    Raises:
        RuntimeError: If no sub-estimators are scheduled for training.

    """
    estimator_params = estimator.get_params()

    params_list = [
        dict(zip(param_grid, param_values))
        for param_values in product(*param_grid.values())
    ]

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    X_train_f = []
    y_train_f = []

    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])

        X_train_f.append(client.scatter(X_train[-1], broadcast=True))
        y_train_f.append(client.scatter(y_train[-1], broadcast=True))

    score_event = Event()
    estimator_score_count = 0
    results = defaultdict(lambda: defaultdict(dict))

    def get_estimator_score_cb(estimator, param_key, split_index):
        """Get a function that scores the given estimator.

        Args:
            estimator (object implementing a `score()` method): Estimator to be
                evaluated.
            param_key (tuple): Tuple identifying the parameters of `estimator` that
                were modified during the grid search.
            split_index (int): Index of the current split.

        Returns:
            callable: Callable with signature (future), where `future.result()`
                contains the trained sub-estimators that will be placed into
                `estimator.estimators_`.

        """

        def construct_and_score(future):
            """Join the sub-estimators and score the resulting estimator.

            Scores will be placed into the global `results` dict.

            Score completion will be signalled using the `score_event` Event.

            Args:
                future (future): `future.result()` contains the trained sub-estimators
                    that will be placed into `estimator.estimators_`.

            """
            nonlocal estimator_score_count

            estimator.estimators_.extend([f for f in future.result()])

            # NOTE: `RandomForestRegressor.predict()` calculates `n_jobs` internally
            # using `joblib.effective_n_jobs()` without considering `n_jobs` from the
            # currently enabled default backend. Therefore, wrapping `score()` in
            # `parallel_backend('threading', n_jobs=local_n_jobs)` only runs on
            # `estimator.n_jobs` threads (in the case where `estimator.n_jobs = None`,
            # this causes the scoring to run on a single thread only), regardless of
            # the value of `local_n_jobs`.

            # Force the use of `local_n_jobs` threads in `score()` (see above).
            with ExitStack() as stack:
                stack.enter_context(
                    temp_sklearn_params(estimator, {"n_jobs": local_n_jobs})
                )
                stack.enter_context(parallel_backend("threading", n_jobs=local_n_jobs))

                results[param_key]["test_score"][split_index] = estimator.score(
                    X_test[split_index], y_test[split_index]
                )
                if return_train_score:
                    results[param_key]["train_score"][split_index] = estimator.score(
                        X_train[split_index], y_train[split_index]
                    )
            estimator_score_count += 1
            score_event.set()

        return construct_and_score

    # Collect all sub-estimator futures for progress monitoring.
    sub_estimator_fs = []

    def estimator_fit_done_callback(futures):
        return futures

    # Task submission progress bar.
    submit_tqdm = tqdm(
        desc="Submitting tasks",
        total=len(params_list) * n_splits,
        disable=not verbose,
        unit="task",
        smoothing=0.01,
    )

    for grid_params in params_list:
        # Hashable version of the param dict.
        param_key = tuple(sorted(grid_params.items()))

        for split_index, (X_t, y_t, X_t_f, y_t_f) in enumerate(
            zip(X_train, y_train, X_train_f, y_train_f)
        ):
            grid_estimator = clone(estimator).set_params(
                **{**estimator_params, **grid_params}
            )
            sub_est_fs = grid_estimator.dask_fit(X_t, y_t, X_t_f, y_t_f, client)

            if not sub_est_fs:
                raise RuntimeError("No sub-estimators were scheduled to be trained.")

            sub_estimator_fs.extend(sub_est_fs)

            # Collate the sub-estimator futures to process them collectively later.
            estimator_future = client.submit(estimator_fit_done_callback, sub_est_fs)
            estimator_future.add_done_callback(
                get_estimator_score_cb(grid_estimator, param_key, split_index)
            )
            submit_tqdm.update()

    submit_tqdm.close()

    def sub_estimator_progress():
        """Progress bar for completed sub-estimators."""
        for f in tqdm(
            as_completed(sub_estimator_fs),
            desc="Sub-estimators",
            total=len(sub_estimator_fs),
            disable=not verbose,
            unit="sub-estimator",
            smoothing=0.01,
            position=1,
        ):
            pass

    sub_estimator_progress_thread = Thread(target=sub_estimator_progress)
    sub_estimator_progress_thread.start()

    # Progress bar for completed estimator scores.
    score_tqdm = tqdm(
        desc="Scoring",
        total=len(params_list) * n_splits,
        disable=not verbose,
        unit="estimator",
        smoothing=0.1,
        position=0,
    )

    while estimator_score_count != len(params_list) * n_splits:
        # Wait for a scoring operation to be completed.
        score_event.wait()
        # Ready the event for the next scoring.
        score_event.clear()
        # Update the progress bar.
        score_tqdm.update(estimator_score_count - score_tqdm.n)

    score_tqdm.close()

    # Join the sub-estimator progress bar thread.
    sub_estimator_progress_thread.join()

    # Collate the scores.
    for estimator_params, param_results in results.items():
        score_keys = ["test_score"]
        if return_train_score:
            score_keys.append("train_score")

        for score_key in score_keys:
            param_results[score_key] = [
                param_results[score_key][i]
                for i in range(len(param_results[score_key]))
            ]

    if refit:
        mean_test_scores = {}
        for estimator_params, param_results in results.items():
            mean_test_scores[estimator_params] = np.mean(param_results["test_score"])
        best_params = dict(max(mean_test_scores, key=lambda k: mean_test_scores[k]))
        refit_estimator = clone(estimator).set_params(**best_params)
        with parallel_backend("dask", scatter=[X, y]):
            refit_estimator.fit(X, y)
        return dict(results), refit_estimator
    return dict(results)


def safe_write(obj, target, prefix=None, suffix=None):
    """Write `obj` to the path `target` via a temporary file using pickle.

    Args:
        obj: Object to pickle.
        target (str or pathlib.Path): Destination path.
        prefix (str or None): String to prepend to the temporary filename.
        suffix (str or None): String to append to the temporary filename.

    """
    with NamedTemporaryFile(
        mode="wb", prefix=prefix, suffix=suffix, delete=False
    ) as tmp_file:
        pickle.dump(obj, tmp_file, -1)
    shutil.move(tmp_file.name, target)


class CachedResults(defaultdict):
    """Nested defaultdict with dict default_factory and caching methods."""

    def __init__(
        self, *, estimator_class=None, n_splits=None, cache_dir=None, **kwargs
    ):
        """Initialise the caching.

        Args:
            estimator_class: Class of the estimator object being fit.
            n_splits (int): Number of CV splits.
            cache_dir (str, pathlib.Path, or None): Directory to save score results
                and fitted estimators in. If `None`, caching is disabled. If not
                `None`, both `estimator_class` and `n_splits` have to be given.

        """
        super().__init__(lambda: defaultdict(dict), **kwargs)
        self.estimator_class = estimator_class
        self.n_splits = n_splits
        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = Path(cache_dir) / estimator_class.__name__ / str(n_splits)
            self.score_file = self.cache_dir / "scores.pkl"
            self.estimator_file = self.cache_dir / "estimators.pkl"
        if self.cache_dir is not None:
            if self.score_file.is_file():
                with open(self.score_file, "rb") as f:
                    self.update(pickle.load(f))
            else:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

    def copy(self):
        copied = type(self)(
            estimator_class=self.estimator_class,
            n_splits=self.n_splits,
            cache_dir=None if self.cache_dir is None else self.cache_dir.parents[1],
        )
        copied.update(self)
        return copied

    def cache(self):
        """Cache the scores."""
        if self.cache_dir is not None:
            safe_write(dict(self), self.score_file, prefix="scores_", suffix=".pkl")

    def _load_estimator_cache(self):
        """Load the estimator cache pickle file.

        Returns:
            dict: Mapping from estimator parameters to fitted parameters.

        """
        if self.cache_dir is None or not self.estimator_file.is_file():
            raise KeyError("Caching is disabled or non-existent.")
        with open(self.estimator_file, "rb") as f:
            return pickle.load(f)

    def get_estimator(self, key):
        """Retrieve a fitted estimator.

        Args:
            key (hashable object): Key identifying the model

        Returns:
            Fitted estimator.

        Raises:
            KeyError: If `key` is not found within the cache or caching is disabled.

        """
        return self._load_estimator_cache()[key]

    def store_estimator(self, key, estimator):
        """Cache a fitted estimator.

        Args:
            key (hashable object): Key identifying the model
            estimator: The fitted estimator.

        """
        if self.cache_dir is not None:
            if self.estimator_file.is_file():
                with open(self.estimator_file, "rb") as f:
                    cached_estimators = pickle.load(f)
            else:
                cached_estimators = {}
            cached_estimators[key] = estimator
            safe_write(
                cached_estimators, self.estimator_file, prefix="models_", suffix=".pkl"
            )

    def collate_scores(self, train_scores=None):
        """Collate results from different splits into combined lists.

        Args:
            train_scores (bool or None): If `True`, return train scores. If `None`,
                return train scores only if they are available.

        Returns:
            dict: Collated results.

        """
        results = dict(self)
        results = deepcopy(results)

        for estimator_params, param_results in results.items():
            score_keys = ["test_score"]
            if train_scores or (
                train_scores is None and "train_score" in param_results
            ):
                score_keys.append("train_score")

            for score_key in score_keys:
                param_results[score_key] = [
                    param_results[score_key][i]
                    for i in range(len(param_results[score_key]))
                ]
        return results

    def get_best_params(self, key="test_score"):
        results = self.collate_scores()
        mean_test_scores = {}
        for estimator_params, param_results in results.items():
            if len(param_results[key]) == self.n_splits:
                mean_test_scores[estimator_params] = np.mean(param_results[key])
        return dict(max(mean_test_scores, key=lambda k: mean_test_scores[k]))


def fit_dask_sub_est_random_search_cv(
    estimator,
    X,
    y,
    param_grid,
    client,
    n_splits=5,
    max_time=None,
    n_iter=10,
    verbose=False,
    refit=True,
    return_train_score=True,
    local_n_jobs=None,
    random_state=None,
    cache_dir=None,
):
    """Carry out a grid search using Dask-adapted scikit-learn estimators.

    The futures returned by `dask_fit()` are tracked and if all sub-estimators
    belonging to an estimator have been trained successfully, the sub-estimators are
    collated and used to perform the scoring.

    Depending on the problem, this method may be slower than using `DaskGridSearchCV`.

    Args:
        estimator (object implementing `dask_fit()` and `score()` methods): Estimator
            to be evaluated.
        X (array-like): Training vector.
        y (array-like): Target relative to `X`.
        param_grid (dict): Dictionary with parameters names (`str`) as keys and
            distributions or lists of parameters to try. Distributions must provide a
            `rvs` method for sampling (such as those from
            `scipy.stats.distributions`). If a list is given, it is sampled uniformly.
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.
        n_splits (int): Number of splits used for `KFold()`.
        max_time (int, str, or None): The maximum time allowed. If an `int` is given,
            the number of seconds is specified. A string may include units, e.g. '1m'
            or '1 minute' for 1 minute. If `None` is given, the number of parameters
            is determined by `n_iter`. If both `max_time` and `n_iter` are given,
            `max_time` will be used.
        n_iter (int or None): The number of parameters to try. Will only be considered
            if `max_time` is `None`. If `None` is given for both `max_time` (or
            `max_time` is 0) and `n_iter`, relevant saved results (if any are found in
            `cache_dir`) will be returned. If `max_time` is `None` and `n_iter` is 0,
            saved results will be returned and `estimator` will be fit using the best
            saved parameters (if any).
        verbose (bool): If True, print out progress information related to the fitting
            of individual sub-estimators and scoring of the resulting estimators.
        refit (bool): If True, fit `estimator` using the best parameters on all of `X`
            and `y`. Time taken here will not be included in budget allocated by
            `max_time`.
        return_train_score (bool): If True, compute training scores.
        local_n_jobs (int): Since scoring has a 'sharedmem' requirement (ie. threading
            backend), parallelisation can be achieved locally using the threading
            backend with `local_n_jobs` threads.
        random_state (None, int, or np.random.RandomState): Random number generator
            state.
        cache_dir (str, pathlib.Path, or None): Directory to save score results and
            fitted estimators (see `refit`) in.

    Returns:
        dict: Dictionary containing the test (and train) scores for individual
            parameters and splits.
        estimator: Only present if `refit` is True. `estimator` fit on `X` and `y`
            using the best parameters found.

    Raises:
        RuntimeError: If no sub-estimators are scheduled for training.

    """
    # Record scores for different parameters.
    results = CachedResults(
        estimator_class=type(estimator), n_splits=n_splits, cache_dir=cache_dir
    )

    if (max_time is None or max_time == 0) and n_iter is None:
        if refit:
            return (
                results.collate_scores(train_scores=return_train_score),
                results.get_estimator(
                    tuple(
                        sorted(
                            clone(estimator)
                            .set_params(**results.get_best_params())
                            .get_params()
                            .items()
                        )
                    )
                ),
            )
        return results.collate_scores(train_scores=return_train_score)

    timeout = Event()
    score_complete = Event()
    timeout_or_tasks = Event()
    timeout_or_score = Event()

    def set_timeout():
        timeout.set()
        timeout_or_tasks.set()
        timeout_or_score.set()

    if max_time is not None:
        n_iter = sys.maxsize
        max_time = parse_timedelta(max_time)
        timeout_thread = Timer(max_time, set_timeout)
        timeout_thread.start()

    random_state = check_random_state(random_state)

    # Get the original parameters.
    estimator_params = estimator.get_params()

    # Create and scatter the train and test dataset indices.
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    X_train_f = []
    y_train_f = []

    # Note that random_state is only used when shuffle=True.
    kf = KFold(n_splits=n_splits)

    def scatter_data():
        # Change broadcast behaviour based on memory availability.
        memory_limits = list(
            w["memory_limit"] for w in client.scheduler_info()["workers"].values()
        )
        if all(isinstance(memory, Number) for memory in memory_limits):
            lowest_memory = min(memory_limits)
            total_required_memory = (n_splits - 1) * X.nbytes
            broadcast = lowest_memory > total_required_memory
        else:
            broadcast = False
        logger.debug(f"Broadcasting data: {broadcast}.")
        for train_index, test_index in tqdm(
            kf.split(X),
            desc="Scattering train and test datasets",
            total=n_splits,
            disable=not verbose,
            unit="split",
        ):
            X_train.append(X[train_index])
            y_train.append(y[train_index])
            X_test.append(X[test_index])
            y_test.append(y[test_index])

            X_train_f.append(client.scatter(X_train[-1], broadcast=broadcast))
            y_train_f.append(client.scatter(y_train[-1], broadcast=broadcast))

    # Define progress bars.

    # Task submission progress bar.
    submit_tqdm = tqdm(
        desc="Submitting tasks",
        total=0,
        disable=not verbose,
        unit="task",
        smoothing=0.01,
        position=0,
    )

    # Progress bar for completed estimator scores.
    score_tqdm = tqdm(
        desc="Scoring",
        total=0,
        disable=not verbose,
        unit="estimator",
        smoothing=0.1,
        position=1,
    )
    score_lock = Lock()

    # Sub-estimator progress bar.
    sub_est_tqdm = tqdm(
        desc="Sub-estimators",
        total=0,
        disable=not verbose,
        unit="sub-estimator",
        smoothing=0.01,
        position=2,
    )
    sub_est_lock = Lock()

    def get_estimator_score_cb(estimator, param_key, split_index):
        """Get a function that scores the given estimator.

        Args:
            estimator (object implementing a `score()` method): Estimator to be
                evaluated.
            param_key (tuple): Tuple identifying the parameters of `estimator` that
                were modified during the grid search.
            split_index (int): Index of the current split.

        Returns:
            callable: Callable with signature (future), where `future.result()`
                contains the trained sub-estimators that will be placed into
                `estimator.estimators_`.

        """

        def construct_and_score(future):
            """Join the sub-estimators and score the resulting estimator.

            Scores will be placed into the global `results` dictionary.

            Args:
                future (future): `future.result()` contains the trained sub-estimators
                    that will be placed into `estimator.estimators_`.

            """
            try:
                estimator.estimators_.extend([f for f in future.result()])
            except CancelledError:
                if not timeout.is_set():
                    raise
                else:
                    # We expect futures to be cancelled.
                    # We do not want to carry out any scoring after the timeout.
                    return

            # NOTE: `RandomForestRegressor.predict()` calculates `n_jobs` internally
            # using `joblib.effective_n_jobs()` without considering `n_jobs` from the
            # currently enabled default backend. Therefore, wrapping `score()` in
            # `parallel_backend('threading', n_jobs=local_n_jobs)` only runs on
            # `estimator.n_jobs` threads (in the case where `estimator.n_jobs = None`,
            # this causes the scoring to run on a single thread only), regardless of
            # the value of `local_n_jobs`.

            # Force the use of `local_n_jobs` threads in `score()` (see above).
            with ExitStack() as stack:
                stack.enter_context(
                    temp_sklearn_params(estimator, {"n_jobs": local_n_jobs})
                )
                stack.enter_context(parallel_backend("threading", n_jobs=local_n_jobs))

                if timeout.is_set():
                    return

                if not (
                    param_key in results
                    and "test_score" in results[param_key]
                    and split_index in results[param_key]["test_score"]
                ):
                    results[param_key]["test_score"][split_index] = estimator.score(
                        X_test[split_index], y_test[split_index]
                    )
                if return_train_score:
                    if timeout.is_set():
                        return

                    if not (
                        param_key in results
                        and "train_score" in results[param_key]
                        and split_index in results[param_key]["train_score"]
                    ):
                        results[param_key]["train_score"][
                            split_index
                        ] = estimator.score(X_train[split_index], y_train[split_index])

            # Cache results.
            results.cache()

            with score_lock:
                score_tqdm.update()
                if score_tqdm.n == score_tqdm.total:
                    score_complete.set()
                    timeout_or_score.set()

        return construct_and_score

    critical_target = math.inf
    critical_target_lock = Lock()
    critical_task_count = Event()

    def update_sub_est_progress(future):
        """Progress bar for completed sub-estimators."""
        nonlocal critical_target
        with sub_est_lock:
            sub_est_tqdm.update()

        # Only test this criterion after the target has been updated.
        if not critical_task_count.is_set():
            with critical_target_lock, sub_est_lock:
                if sub_est_tqdm.n > critical_target:
                    critical_task_count.set()
                    timeout_or_tasks.set()

    def estimator_fit_done_callback(futures):
        """Used to collate sub-estimators belonging to a single estimator."""
        return futures

    # Collect futures to cancel them all later.
    all_futures = []

    def grid_params_split():
        """Iterator over parameters and splits."""
        all_lists = all(not hasattr(v, "rvs") for v in param_grid.values())
        parameter_sampler = None
        if all_lists:
            full_param_grid = ParameterGrid(param_grid)
            if n_iter > len(full_param_grid):
                # If no combinations are excluded, make sure that they are sampled
                # randomly (unlike ParameterSampler) in case the timeout occurs.
                parameter_sampler = (
                    full_param_grid[i]
                    for i in random_state.permutation(np.arange(len(full_param_grid)))
                )
        if parameter_sampler is None:
            parameter_sampler = ParameterSampler(
                param_grid, n_iter, random_state=random_state
            )

        for grid_params in parameter_sampler:
            for split_index in range(n_splits):
                yield grid_params, split_index

    def submit_tasks():
        """Submit a set of sub-estimators.

        This is done for each parameter combination and split separately.

        Yields:
            int: The number of submitted sub-estimators.
            float: The time taken to submit the tasks.

        """
        scattered = False
        for grid_params, split_index in grid_params_split():
            # Hashable version of the param dict.
            param_key = tuple(sorted(grid_params.items()))

            score_keys = ["test_score"]
            if return_train_score:
                score_keys.append("train_score")
            if all(
                param_key in results
                and score_key in results[param_key]
                and split_index in results[param_key][score_key]
                for score_key in score_keys
            ):
                # If everything is already cached, signal that scoring is complete.
                with score_lock:
                    if score_tqdm.n == score_tqdm.total:
                        score_complete.set()
                        timeout_or_score.set()
                # Skip cached results.
                yield 0
                continue

            # Only scatter the data (once) if results are not cached already.
            if not scattered:
                scatter_data()
                scattered = True

            X_t, y_t, X_t_f, y_t_f = map(
                itemgetter(split_index), (X_train, y_train, X_train_f, y_train_f)
            )

            if timeout.is_set():
                # Do not submit additional tasks after the timeout.
                logger.warning("Timeout encountered. Ceasing task submission.")
                break

            grid_estimator = clone(estimator).set_params(
                **{**estimator_params, **grid_params}
            )
            grid_estimator_params = grid_estimator.get_params()
            if "n_estimators" in grid_estimator_params:
                submit_tqdm.total += grid_estimator_params["n_estimators"]
                submit_tqdm.update(0)

            # Submit the fitting of the sub estimators.
            sub_est_fs = grid_estimator.dask_fit(X_t, y_t, X_t_f, y_t_f, client)

            if not sub_est_fs:
                raise RuntimeError("No sub-estimators were scheduled to be trained.")

            # Update progress meter upon task completion.
            for sub_est_f in sub_est_fs:
                sub_est_f.add_done_callback(update_sub_est_progress)

            # Keep track of the sub-estimator tasks and their number.
            with sub_est_lock:
                sub_est_tqdm.total += len(sub_est_fs)
                sub_est_tqdm.update(0)

            # Increment the outstanding number of score operations.
            with score_lock:
                score_complete.clear()
                timeout_or_score.clear()
                score_tqdm.total += 1
                score_tqdm.update(0)

            # Collate the sub-estimator futures to process them collectively later.
            estimator_future = client.submit(estimator_fit_done_callback, sub_est_fs)
            # Add the estimator scoring callback.
            estimator_future.add_done_callback(
                get_estimator_score_cb(grid_estimator, param_key, split_index)
            )

            all_futures.extend(sub_est_fs)
            all_futures.append(estimator_future)

            submit_tqdm.update(len(sub_est_fs))

            yield len(sub_est_fs)

    # Submit tasks.
    submission_start = time.time()
    n_avg = 10
    avg_fit_times = deque(maxlen=n_avg)
    avg_weights = np.exp(-6 * np.linspace(0, 1, n_avg))
    for n_sub_est in submit_tasks():
        if n_sub_est == 0:
            # As the results were already cached, try to submit another batch.
            continue
        total_cores = sum(client.ncores().values())  # This may change with time.
        target_tasks = 2 * total_cores  # Prefetching of tasks.
        with sub_est_lock:
            currently_active = sub_est_tqdm.total - sub_est_tqdm.n
        # If we are currently running more tasks than our target, determine how long
        # to wait before submitting additional tasks. Otherwise submit more tasks
        # immediately.
        if currently_active > target_tasks:
            # Determine the number of tasks to complete before new tasks are
            # scheduled.
            with critical_target_lock, sub_est_lock:
                n_done = max((sub_est_tqdm.n, 1))

                # Average time per fit.
                last_avg_fit_time = (time.time() - submission_start) / n_done
                if len(avg_fit_times) == 0:
                    avg_fit_times.extend([last_avg_fit_time] * n_avg)
                else:
                    avg_fit_times.appendleft(last_avg_fit_time)

                # Critical number of active tasks. Take into account average task
                # duration and a margin of 2 seconds.

                # Average the most recent average fit times (ignoring weighting by
                # number of samples).
                avg_fit_time = np.average(avg_fit_times, weights=avg_weights)
                critical_target = math.floor(
                    sub_est_tqdm.total - target_tasks - (2 / avg_fit_time)
                )
                # Reset the task count event to signal the updated target.
                critical_task_count.clear()
                if critical_target <= sub_est_tqdm.n:
                    # If enough tasks have already finished, schedule more.
                    critical_task_count.set()
                    timeout_or_tasks.set()

            # Wait for the desired number of tasks to finish.
            while not (critical_task_count.is_set() or timeout.is_set()):
                timeout_or_tasks.wait()
                timeout_or_tasks.clear()

    if n_iter != 0:
        if max_time is None:
            # If `n_iter` only was used, wait for the desired number of parameters to
            # be processed.
            score_complete.wait()
        else:
            # Wait for the timeout or scoring to finish.
            while not (score_complete.is_set() or timeout.is_set()):
                timeout_or_score.wait()
            # In case scoring finished before the timeout, cancel it.
            timeout_thread.cancel()

    for progress_tqdm in (submit_tqdm, score_tqdm, sub_est_tqdm):
        progress_tqdm.close()

    # Cancel futures.
    if all_futures:
        client.cancel(all_futures, force=True)

    if refit:
        refit_estimator = clone(estimator).set_params(**results.get_best_params())
        refit_est_key = tuple(sorted(refit_estimator.get_params().items()))
        try:
            refit_estimator = results.get_estimator(refit_est_key)
            logger.debug(f"Loaded estimator with params {refit_est_key} from cache.")
        except KeyError:
            logger.debug(
                f"Estimator with params {refit_est_key} could not be found. Fitting "
                "now."
            )
            with parallel_backend("dask", scatter=[X, y]):
                refit_estimator.fit(X, y)
            results.store_estimator(refit_est_key, refit_estimator)
        return results.collate_scores(), refit_estimator
    return results.collate_scores()


class DaskGridSearchCV(GridSearchCV):
    def dask_fit(self, client, X, y=None, groups=None, **fit_params):
        """Run a distributed fit with all sets of parameters.

        Individual `fit()` and `score()` calls are carried out on single workers.
        Workers need to specify the resource 'threads', e.g. '--resources threads=32',
        to specify the number of threads available for model fitting and scoring per
        worker, regardless of the number of physical cores Dask believes are available
        (additional cores may have been requested in a job submission script, for
        example). This number of threads is then reflected in the estimator's `n_jobs`
        parameter to achieve parallelism on each worker.

        Depending on the problem, this method may be slower than
        `fit_dask_sub_est_grid_search_cv()`.

        Only use this method if individual calls to `fit()` and `score()` can both be
        parallelised (and to the same extent). Otherwise other approaches are more
        applicable, such as dask-ml's `GridSearchCV`.

        Changed parameter semantics:

            `self.verbose` is interpreted to be either False or True,
            displaying a progress bar when True.

            `self.pre_dispatch` is ignored.

        This function was adapted from sklearn.model_selection._validation (28-05-2020).

        Parameters
        ----------
        client : `dask.distributed.Client`
            Dask Client used to submit tasks to the scheduler.
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Raises
        ------
        RuntimeError
            If the Dask workers associated with `client` do not specify the 'threads'
            resource.
        RuntimeError
            If the 'threads' resources specified by the Dask workers associated with
            `client` do not match.

        """
        n_jobs = common_worker_threads(client)

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring
        )

        if self.multimetric_:
            if self.refit is not False and (
                not isinstance(self.refit, six.string_types)
                or
                # This will work for both dict / list (tuple)
                self.refit not in scorers
            ):
                raise ValueError(
                    "For multi-metric scoring, the parameter "
                    "refit must be set to a scorer key "
                    "to refit an estimator with the best "
                    "parameter setting on the whole data and "
                    "make the best_* attributes "
                    "available for that metric. If this is not "
                    "needed, refit should be set to False "
                    "explicitly. %r was passed." % self.refit
                )
            else:
                refit_metric = self.refit
        else:
            refit_metric = "score"

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results_container = [{}]

        all_candidate_params = []
        all_out = []

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                print(
                    "Fitting {0} folds for each of {1} candidates,"
                    " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            X_fut = client.scatter(X, broadcast=True)
            y_fut = client.scatter(y, broadcast=True)

            out_fs = [
                client.submit(
                    _fit_and_score,
                    clone(base_estimator).set_params(n_jobs=n_jobs),
                    X_fut,
                    y_fut,
                    train=train,
                    test=test,
                    parameters=parameters,
                    resources={"threads": n_jobs},
                    **fit_and_score_kwargs,
                )
                for parameters, (train, test) in product(
                    candidate_params, cv.split(X, y, groups)
                )
            ]

            # Get a progress bar of completed futures.
            for f in tqdm(
                as_completed(out_fs),
                total=len(out_fs),
                unit="fit",
                desc="Carrying out grid search",
                smoothing=0.05,
                disable=not self.verbose,
            ):
                pass

            # Append the finished calculations to `out` in order.
            out = [out_f.result() for out_f in out_fs]

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            results_container[0] = self._format_results(
                all_candidate_params, scorers, n_splits, all_out
            )
            return results_container[0]

        self._run_search(evaluate_candidates)

        results = results_container[0]

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = results["params"][self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


def dask_fit_loco(
    estimator, X_train, y_train, client, leave_out, local_n_jobs=None, verbose=False
):
    """Simple LOCO feature importances.

    Args:
        estimator (object implementing `dask_fit()` and `score()` methods): Estimator
            to be evaluated.
        train_X (pandas DataFrame): DataFrame containing the training data.
        train_y (pandas Series or array-like): Target data.
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.
        leave_out (iterable of column names): Column names to exclude. An empty string
            indicates that all columns should be used (baseline).
        local_n_jobs (int): Since scoring has a 'sharedmem' requirement (ie. threading
            backend), parallelisation can be achieved locally using the threading
            backend with `local_n_jobs` threads.
        verbose (bool): If True, print out progress information related to the fitting
            of individual sub-estimators and scoring of the resulting estimators.

    Returns:
        mse: Mean squared error of the training set predictions.

    """
    score_event = Event()
    estimator_score_count = 0
    results = defaultdict(lambda: defaultdict(dict))

    y_train = np.asarray(y_train)
    y_train_f = client.scatter(y_train, broadcast=True)

    def get_estimator_score_cb(estimator, column):
        """Get a function that scores the given estimator.

        Args:
            estimator (object implementing a `socre()` method): Estimator to be
                evaluated.
            column (str): Column to discard.

        Returns:
            callable: Callable with signature (future), where `future.result()`
                contains the trained sub-estimators that will be placed into
                `estimator.estimators_`.

        """

        def construct_and_score(future):
            """Join the sub-estimators and score the resulting estimator.

            Scores will be placed into the global `results` dict.

            Score completion will be signalled using the `score_event` Event.

            Args:
                future (future): `future.result()` contains the trained sub-estimators
                    that will be placed into `estimator.estimators_`.

            """
            nonlocal estimator_score_count

            estimator.estimators_.extend([f for f in future.result()])

            # NOTE: `RandomForestRegressor.predict()` calculates `n_jobs` internally
            # using `joblib.effective_n_jobs()` without considering `n_jobs` from the
            # currently enabled default backend. Therefore, wrapping `score()` in
            # `parallel_backend('threading', n_jobs=local_n_jobs)` only runs on
            # `estimator.n_jobs` threads (in the case where `estimator.n_jobs = None`,
            # this causes the scoring to run on a single thread only), regardless of
            # the value of `local_n_jobs`.

            # Force the use of `local_n_jobs` threads in `score()` (see above).
            with ExitStack() as stack:
                stack.enter_context(
                    temp_sklearn_params(estimator, {"n_jobs": local_n_jobs})
                )
                stack.enter_context(parallel_backend("threading", n_jobs=local_n_jobs))

                sel_X_train = np.asarray(
                    X_train[[col for col in X_train.columns if col != column]]
                )
                y_pred = estimator.predict(sel_X_train)
                results[column]["score"] = r2_score(y_true=y_train, y_pred=y_pred)
                results[column]["mse"] = mean_squared_error(
                    y_true=y_train, y_pred=y_pred
                )
            estimator_score_count += 1
            score_event.set()

        return construct_and_score

    # Collect all sub-estimator futures for progress monitoring.
    sub_estimator_fs = []

    def estimator_fit_done_callback(futures):
        return futures

    # Task submission progress bar.
    submit_tqdm = tqdm(
        desc="Submitting tasks",
        total=len(leave_out),
        disable=not verbose,
        unit="task",
        smoothing=0.01,
    )

    for column in leave_out:
        grid_estimator = clone(estimator)
        sel_X_train = np.asarray(
            X_train[[col for col in X_train.columns if col != column]]
        )
        sel_X_train_f = client.scatter(sel_X_train)
        # TODO: Exclude column on the worker to avoid storing redundant copies that
        # only differ in one column each.
        sub_est_fs = grid_estimator.dask_fit(
            sel_X_train, y_train, sel_X_train_f, y_train_f, client
        )

        if not sub_est_fs:
            raise RuntimeError("No sub-estimators were scheduled to be trained.")

        sub_estimator_fs.extend(sub_est_fs)

        # Collate the sub-estimator futures to process them collectively later.
        estimator_future = client.submit(estimator_fit_done_callback, sub_est_fs)
        estimator_future.add_done_callback(
            get_estimator_score_cb(grid_estimator, column)
        )
        submit_tqdm.update()

    submit_tqdm.close()

    def sub_estimator_progress():
        """Progress bar for completed sub-estimators."""
        for f in tqdm(
            as_completed(sub_estimator_fs),
            desc="Sub-estimators",
            total=len(sub_estimator_fs),
            disable=not verbose,
            unit="sub-estimator",
            smoothing=0.01,
            position=1,
        ):
            pass

    sub_estimator_progress_thread = Thread(target=sub_estimator_progress)
    sub_estimator_progress_thread.start()

    # Progress bar for completed estimator scores.
    score_tqdm = tqdm(
        desc="Scoring",
        total=len(leave_out),
        disable=not verbose,
        unit="estimator",
        smoothing=0.1,
        position=0,
    )

    while estimator_score_count != len(leave_out):
        # Wait for a scoring operation to be completed.
        score_event.wait()
        # Ready the event for the next scoring.
        score_event.clear()
        # Update the progress bar.
        score_tqdm.update(estimator_score_count - score_tqdm.n)

    score_tqdm.close()

    # Join the sub-estimator progress bar thread.
    sub_estimator_progress_thread.join()

    return results


def dask_fit_combinations(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    client,
    feature_combinations,
    local_n_jobs=None,
    verbose=False,
    cache_dir=None,
):
    """Fit all combinations of input features.

    Args:
        estimator (object implementing `dask_fit()` and `score()` methods): Estimator
            to be evaluated.
        train_X (pandas DataFrame): Training data.
        train_y (pandas Series or array-like): Training target data.
        test_X (pandas DataFrame): Test data.
        test_y (pandas Series or array-like): Test target data.
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.
        feature_combinations (iterable of list of str): Feature combinations to fit.
        local_n_jobs (int): Since scoring has a 'sharedmem' requirement (ie. threading
            backend), parallelisation can be achieved locally using the threading
            backend with `local_n_jobs` threads.
        verbose (bool): If True, print out progress information related to the fitting
            of individual sub-estimators and scoring of the resulting estimators.
        cache_dir (str, pathlib.Path, or None): Directory to save score results in.

    Returns:
        mse: Mean squared error of the training set predictions.

    """
    feature_combinations = list(feature_combinations)

    if cache_dir is not None:
        cache_file = (
            Path(cache_dir)
            / "combinations"
            / type(estimator).__name__
            / str(len(feature_combinations[0]))
            / "scores.pkl"
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        cache_file = None

    def read_scores():
        if cache_dir is None or not cache_file.is_file():
            return defaultdict(dict)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def write_scores(scores):
        if cache_dir is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(scores, f, -1)

    scores = read_scores()
    if len(scores) == len(feature_combinations):
        return scores

    y_t_f = client.scatter(y_train, broadcast=True)

    score_complete = Event()

    # Define progress bars.

    # Task submission progress bar.
    submit_tqdm = tqdm(
        desc="Submitting tasks",
        total=0,
        disable=not verbose,
        unit="task",
        smoothing=0.01,
        position=0,
    )

    # Progress bar for completed estimator scores.
    score_tqdm = tqdm(
        desc="Scoring",
        total=0,
        disable=not verbose,
        unit="estimator",
        smoothing=0.1,
        position=1,
    )
    score_lock = Lock()

    # Sub-estimator progress bar.
    sub_est_tqdm = tqdm(
        desc="Sub-estimators",
        total=0,
        disable=not verbose,
        unit="sub-estimator",
        smoothing=0.01,
        position=2,
    )
    sub_est_lock = Lock()

    def get_estimator_score_cb(estimator, combination):
        """Get a function that scores the given estimator.

        Args:
            estimator (object implementing a `score()` method): Estimator to be
                evaluated.
            combination (tuple of str): Column names.

        Returns:
            callable: Callable with signature (future), where `future.result()`
                contains the trained sub-estimators that will be placed into
                `estimator.estimators_`.

        """

        def construct_and_score(future):
            """Join the sub-estimators and score the resulting estimator.

            Scores will be placed into the global `scores` dict.

            Score completion will be signalled using the `score_event` Event.

            Args:
                future (future): `future.result()` contains the trained sub-estimators
                    that will be placed into `estimator.estimators_`.

            """
            estimator.estimators_.extend([f for f in future.result()])

            # NOTE: `RandomForestRegressor.predict()` calculates `n_jobs` internally
            # using `joblib.effective_n_jobs()` without considering `n_jobs` from the
            # currently enabled default backend. Therefore, wrapping `score()` in
            # `parallel_backend('threading', n_jobs=local_n_jobs)` only runs on
            # `estimator.n_jobs` threads (in the case where `estimator.n_jobs = None`,
            # this causes the scoring to run on a single thread only), regardless of
            # the value of `local_n_jobs`.

            # Force the use of `local_n_jobs` threads in `score()` (see above).
            with ExitStack() as stack:
                stack.enter_context(
                    temp_sklearn_params(estimator, {"n_jobs": local_n_jobs})
                )
                stack.enter_context(parallel_backend("threading", n_jobs=local_n_jobs))

                y_test_pred = estimator.predict(X_test[list(combination)].to_numpy())
                scores[combination]["test_score"] = {
                    "r2": r2_score(y_test, y_test_pred),
                    "mse": mean_squared_error(y_test, y_test_pred),
                }

                y_train_pred = estimator.predict(X_train[list(combination)].to_numpy())
                scores[combination]["train_score"] = {
                    "r2": r2_score(y_train, y_train_pred),
                    "mse": mean_squared_error(y_train, y_train_pred),
                }

            # Cache results.
            write_scores(scores)

            with score_lock:
                score_tqdm.update()
                if score_tqdm.n == score_tqdm.total:
                    score_complete.set()

        return construct_and_score

    critical_target = math.inf
    critical_target_lock = Lock()
    critical_task_count = Event()

    def update_sub_est_progress(future):
        """Progress bar for completed sub-estimators."""
        nonlocal critical_target
        with sub_est_lock:
            sub_est_tqdm.update()

        # Only test this criterion after the target has been updated.
        if not critical_task_count.is_set():
            with critical_target_lock, sub_est_lock:
                if sub_est_tqdm.n > critical_target:
                    critical_task_count.set()

    def estimator_fit_done_callback(futures):
        """Used to collate sub-estimators belonging to a single estimator."""
        return futures

    # Collect futures to cancel them all later.
    all_futures = []

    def submit_tasks():
        """Submit a set of sub-estimators.

        This is done for each parameter combination and split separately.

        Yields:
            int: The number of submitted sub-estimators.
            float: The time taken to submit the tasks.

        """
        for combination in feature_combinations:
            if all(
                score_key in scores[combination]
                for score_key in ["test_score", "train_score"]
            ):
                # If everything is already cached, signal that scoring is complete.
                with score_lock:
                    if score_tqdm.n == score_tqdm.total:
                        score_complete.set()
                # Skip cached results.
                yield 0
                continue

            comb_estimator = clone(estimator)

            # Scatter the data.
            X_t = X_train[list(combination)].to_numpy()
            X_t_f = client.scatter(X_t)

            # Submit the fitting of the sub estimators.
            sub_est_fs = comb_estimator.dask_fit(X_t, y_train, X_t_f, y_t_f, client)

            if not sub_est_fs:
                raise RuntimeError("No sub-estimators were scheduled to be trained.")

            submit_tqdm.total += comb_estimator.get_params()["n_estimators"]
            submit_tqdm.update(0)

            # Update progress meter upon task completion.
            for sub_est_f in sub_est_fs:
                sub_est_f.add_done_callback(update_sub_est_progress)

            # Keep track of the sub-estimator tasks and their number.
            with sub_est_lock:
                sub_est_tqdm.total += len(sub_est_fs)
                sub_est_tqdm.update(0)

            # Increment the outstanding number of score operations.
            with score_lock:
                score_complete.clear()
                score_tqdm.total += 1
                score_tqdm.update(0)

            # Collate the sub-estimator futures to process them collectively later.
            estimator_future = client.submit(estimator_fit_done_callback, sub_est_fs)
            # Add the estimator scoring callback.
            estimator_future.add_done_callback(
                get_estimator_score_cb(comb_estimator, combination)
            )

            all_futures.extend(sub_est_fs)
            all_futures.append(estimator_future)

            submit_tqdm.update(len(sub_est_fs))

            yield len(sub_est_fs)

    # Submit tasks.
    submission_start = time.time()
    n_avg = 10
    avg_fit_times = deque(maxlen=n_avg)
    avg_weights = np.exp(-6 * np.linspace(0, 1, n_avg))
    for n_sub_est in submit_tasks():
        if n_sub_est == 0:
            # As the results were already cached, try to submit another batch.
            continue
        total_cores = sum(client.ncores().values())  # This may change with time.
        target_tasks = 2 * total_cores  # Prefetching of tasks.
        with sub_est_lock:
            currently_active = sub_est_tqdm.total - sub_est_tqdm.n
        # If we are currently running more tasks than our target, determine how long
        # to wait before submitting additional tasks. Otherwise submit more tasks
        # immediately.
        if currently_active > target_tasks:
            # Determine the number of tasks to complete before new tasks are
            # scheduled.
            with critical_target_lock, sub_est_lock:
                n_done = max((sub_est_tqdm.n, 1))

                # Average time per fit.
                last_avg_fit_time = (time.time() - submission_start) / n_done
                if len(avg_fit_times) == 0:
                    avg_fit_times.extend([last_avg_fit_time] * n_avg)
                else:
                    avg_fit_times.appendleft(last_avg_fit_time)

                # Critical number of active tasks. Take into account average task
                # duration and a margin of 2 seconds.

                # Average the most recent average fit times (ignoring weighting by
                # number of samples).
                avg_fit_time = np.average(avg_fit_times, weights=avg_weights)
                critical_target = math.floor(
                    sub_est_tqdm.total - target_tasks - (2 / avg_fit_time)
                )
                # Reset the task count event to signal the updated target.
                critical_task_count.clear()
                if critical_target <= sub_est_tqdm.n:
                    # If enough tasks have already finished, schedule more.
                    critical_task_count.set()

            # Wait for the desired number of tasks to finish.
            critical_task_count.wait()

    score_complete.wait()

    for progress_tqdm in (submit_tqdm, score_tqdm, sub_est_tqdm):
        progress_tqdm.close()

    # Cancel futures.
    if all_futures:
        client.cancel(all_futures, force=True)

    return scores
