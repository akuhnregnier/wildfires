# -*- coding: utf-8 -*-
from itertools import product
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
from tqdm import tqdm

__all__ = ("DaskRandomForestRegressor", "FitDaskRFGridSearchCV")


class DaskRandomForestRegressor(RandomForestRegressor):
    def dask_fit(self, X, y, X_f, y_f, client, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Note that this implementation returns futures instead of `self`, which is most
        useful when fitting a series of trees in parallel, eg. using Dask, as is done
        in `FitDaskRFGridSearchCV` which `dask_fit()` was written for.

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


def FitDaskRFGridSearchCV(
    regr,
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
    """Carry out a grid search using the Dask random forest regressor.

    The futures returned by `dask_fit` are tracked and if a whole forest has been
    trained successfully, all relevant trained trees are collected and used to perform
    the scoring.

    Args:
        regr (`DaskRandomForestRegressor`): An instance of `DaskRandomForestRegressor`
            that was initialised with parameters not specified in `param_grid`. These
            parameters will be augmented with those in `param_grid`.
        X (array-like): Training vector.
        y (array-like): Target relative to `X`.
        n_splits (int): Number of splits used for `KFold()`.
        param_grid (dict):
        client (`dask.distributed.Client`): Dask Client used to submit tasks to the
            scheduler.
        verbose (bool): If True, print out progress information related to the fitting
            of individual trees and scoring of the resulting random forest regressors.
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

    """

    params = regr.get_params()

    rf_params_list = [
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

    rf_skel = {}
    for rf_params in rf_params_list:
        params.update(rf_params)

        # Hashable version of the param dict.
        param_key = tuple(sorted(rf_params.items()))
        rf_skel[param_key] = {}

        for split_index, (X_t, y_t, X_t_f, y_t_f) in enumerate(
            zip(X_train, y_train, X_train_f, y_train_f)
        ):
            fit_rf = clone(regr).set_params(**params)
            tree_fs = fit_rf.dask_fit(X_t, y_t, X_t_f, y_t_f, client)

            if not tree_fs:
                raise RuntimeError("No trees were scheduled to be trained.")

            # When all tree futures are done, we can start scoring this RF, after collecting
            # the trees into the parent RF estimator. This is done later.
            rf_skel[param_key][split_index] = {"rf": fit_rf, "tree_fs": tree_fs}

    # Create a list of all tree futures to iterate over.
    tree_fs = [
        f
        for param_results in rf_skel.values()
        for split_results in param_results.values()
        for f in split_results["tree_fs"]
    ]

    # Get a progress bar of completed futures.
    for f in tqdm(
        as_completed(tree_fs),
        total=len(tree_fs),
        unit="trees",
        desc="Training RF trees for different parameters and splits",
        disable=not verbose,
    ):
        # Collect trained trees if all the trees for a RF are done.
        for params, param_results in rf_skel.items():
            for split_index, split_results in param_results.items():
                if "tree_fs" in split_results and all(
                    f.done() for f in split_results["tree_fs"]
                ):
                    rf = split_results["rf"]

                    rf.estimators_.extend(
                        [f.result() for f in split_results["tree_fs"]]
                    )

                    # Delete the futures from the dict to signal that we have
                    # processed them.
                    split_results.clear()

                    with parallel_backend("threading", n_jobs=local_n_jobs):
                        split_results["test_score"] = rf.score(
                            X_test[split_index], y_test[split_index]
                        )
                        if return_train_score:
                            split_results["train_score"] = rf.score(
                                X_train[split_index], y_train[split_index]
                            )

    # Collate the scores.
    for params, param_results in rf_skel.items():
        test_scores = []
        if return_train_score:
            train_scores = []
        for split_index, split_results in param_results.items():
            if "test_score" not in split_results or (
                return_train_score and "train_score" not in split_results
            ):
                raise RuntimeError(
                    f"Scoring failed for parameters: {params} for split {split_index}."
                )
            test_scores.append(split_results["test_score"])
            if return_train_score:
                train_scores.append(split_results["train_score"])

        # Remove the individual entries in favour of the aggregated ones below.
        param_results.clear()

        param_results["test_scores"] = test_scores
        if return_train_score:
            param_results["train_scores"] = train_scores

    if refit:
        mean_test_scores = {}
        for params, param_results in rf_skel.items():
            mean_test_scores[params] = np.mean(param_results["test_scores"])
        best_params = dict(max(mean_test_scores, key=lambda k: mean_test_scores[k]))
        refit_rf = clone(regr).set_params(**best_params)
        with parallel_backend("dask", scatter=[X, y]):
            refit_rf.fit(X, y)
        return rf_skel, refit_rf
    return rf_skel
