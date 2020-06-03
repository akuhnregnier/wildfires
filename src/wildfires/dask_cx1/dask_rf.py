# -*- coding: utf-8 -*-
import time
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
from threading import Event, Thread
from warnings import warn

import numpy as np
from dask.distributed import as_completed
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
from sklearn.model_selection import KFold
from sklearn.model_selection._search import (
    GridSearchCV,
    _check_multimetric_scoring,
    _fit_and_score,
    check_cv,
    indexable,
    is_classifier,
)
from tqdm.auto import tqdm

__all__ = (
    "DaskRandomForestRegressor",
    "fit_dask_sub_est_grid_search_cv",
    "temp_sklearn_params",
    "DaskGridSearchCV",
)


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
        param_grid (dict):
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.
        verbose (bool): If True, print out progress information related to the fitting
            of individual sub-estimators and scoring of the resulting estimators.
        refit (bool): If True, fit `regr` using the best parameters on all of `X` and
            `y`.
        return_train_score (bool): If True, compute training scores.
        local_n_jobs (int): Since scoring has a 'sharedmem' requirement (ie. threading
            backend), parallelisation can be achieved locally using the threading
            backend with `local_n_jobs` threads.

    Returns:
        dict: Dictionary containing the test (and train) scores for individual
            parameters and splits.
        regr: Only present if `refit` is True. `regr` fit on `X` and `y` using the
            best parameters found.

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

        Returns:
            callable: Callable with signature (future), where `future.result()`
                contains the trained sub-estimators that will be placed into
                `estimator.estimators_`.
            param_key (tuple): Tuple identifying the parameters of `estimator` that
                were modified during the grid search.
            split_index (int): Index of the current split.

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
            with temp_sklearn_params(estimator, {"n_jobs": local_n_jobs}), (
                parallel_backend("threading", n_jobs=local_n_jobs),
            ):
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
        return results, refit_estimator
    return results


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
        all_worker_resources = [
            worker["resources"]
            for worker in client.scheduler_info()["workers"].values()
        ]
        if not all("threads" in resources for resources in all_worker_resources):
            raise RuntimeError(
                "Expected all workers to specify the 'threads' resource, but got "
                f"{all_worker_resources}."
            )

        all_worker_threads = [resource["threads"] for resource in all_worker_resources]
        if not all(threads == all_worker_threads[0] for threads in all_worker_threads):
            raise RuntimeError(
                "Expected all workers to have the same number of threads, but got "
                f"{all_worker_threads}."
            )
        n_jobs = all_worker_threads[0]

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
