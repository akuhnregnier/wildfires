# -*- coding: utf-8 -*-
"""Caching of results.

Note that dataframe column name changes may not trigger recalculations.

Due to the custom implementation of lazy proxied objects being returned from the
proxy Joblib backend, all cached functions within this module should be decorated
exclusively with the `cache` decorator defined here. To minimise the possibility of a
Proxy object being realised, as many functions as possible should be cached (at the
expense of storage, of course).

Calling repr() on Proxy objects is fine, but calling str() will realise them
(i.e. call the factory function), so e.g. bare print() statements should not be used.

"""
import logging
import os
from copy import deepcopy
from functools import partial, reduce, wraps
from inspect import signature
from operator import add, attrgetter

import joblib
from joblib import Memory

from ..configuration import DATA_DIR, data_is_available
from ..exceptions import NotCachedError
from .cloudpickle_backend import register_backend as register_cloudpickle_backend
from .hashing import CodeObj
from .iris_backend import register_backend as register_iris_backend
from .proxy_backend import HASHES_ONLY, Factory, HashProxy
from .proxy_backend import register_backend as register_proxy_backend
from .same_call import extract_uniform_args_kwargs

logger = logging.getLogger(__name__)


__all__ = (
    "IN_STORE",
    "mark_dependency",
    "get_memory",
    "IrisMemory",
    "CloudpickleMemory",
    "ProxyMemory",
)

# Sentinel value used to denote calls that are already cached.
IN_STORE = object()


class MemoryMixin:
    @property
    def get_hash(self):
        return self.store_backend.get_hash

    @property
    def location(self):
        return self.store_backend.location

    @property
    def store_backend(self):
        return self.memory.store_backend

    def cache(self, func=None, dependencies=(), ignore=None):
        """A cached function with limited MaskedArray support.

        The added method `check_in_store()` will be added and may be used to determine
        whether a given call is already cached. This method will return `IN_STORE` if the
        call is already cached, or raise a `NotCachedError` otherwise.

        Args:
            func (callable): Function to be cached.
            dependencies (tuple of callable): Other functions the cached function depends
                on. If any of these functions change from one run to the next, the cache
                will be invalidated.
            ignore (iterable of str or None): Arguments to ignore when computing the
                argument hash. This means that changes to those arguments will not cause
                the cache to become invalidated.

        Returns:
            callable: The cached function with added `check_in_store()` method.

        """
        if func is None:
            return partial(
                self.cache,
                dependencies=dependencies,
                ignore=ignore,
            )

        assert callable(func)

        # Update dependencies to enable chaining of dependencies.
        dependencies = (
            *dependencies,
            *reduce(
                add,
                map(
                    tuple,
                    map(
                        default_attrgetter("_orig_func._dependencies", default=()),
                        dependencies,
                    ),
                ),
                (),
            ),
        )

        def _inner_placeholder(hashed, args, kwargs):
            return func(*args, **kwargs)

        _inner_placeholder.__name__ = func.__name__

        cached_inner = self.memory.cache(ignore=["args", "kwargs"])(_inner_placeholder)

        def bound_get_hashed(*orig_args, **orig_kwargs):
            return _get_hashed(
                func,
                *orig_args,
                dependencies=dependencies,
                hash_func=self.get_hash,
                ignore=ignore,
                **orig_kwargs,
            )

        @wraps(func)
        def cached_func(*orig_args, **orig_kwargs):
            hashed = bound_get_hashed(*orig_args, **orig_kwargs)
            return cached_inner(hashed, orig_args, orig_kwargs)

        def check_in_store(*args, **kwargs):
            """Check whether a given call to the cached function is already cached.

            Args:
                args, kwargs: Arguments to check.

            Returns:
                IN_STORE: If the given call was found in the cache.

            Raises:
                NotCachedError: If the given call was not found in the cache.

            """
            output_ids = cached_inner._get_output_identifiers(
                bound_get_hashed(*args, **kwargs), args, kwargs
            )
            if not cached_inner.store_backend.contains_item(output_ids):
                raise NotCachedError(f"The given call is not cached: {output_ids}")
            return IN_STORE

        cached_func.check_in_store = check_in_store
        cached_func._orig_func = func
        cached_func._orig_func._dependencies = dependencies

        return cached_func


class IrisMemory(MemoryMixin):
    def __init__(self, location, **kwargs):
        register_iris_backend()
        self.memory = get_memory(location, backend="iris", **kwargs)


class CloudpickleMemory(MemoryMixin):
    def __init__(self, location, **kwargs):
        register_cloudpickle_backend()
        self.memory = get_memory(location, backend="cloudpickle", **kwargs)


class ProxyMemory(MemoryMixin):
    def __init__(self, location, **kwargs):
        register_proxy_backend()
        self.memory = get_memory(location, backend="proxy", **kwargs)

    def cache(self, func=None, dependencies=(), ignore=None, save_hashes_only=False):
        """A cached function with limited MaskedArray support.

        The added method `check_in_store()` will be added and may be used to determine
        whether a given call is already cached. This method will return `IN_STORE` if the
        call is already cached, or raise a `NotCachedError` otherwise.

        Args:
            func (callable): Function to be cached.
            dependencies (tuple of callable): Other functions the cached function depends
                on. If any of these functions change from one run to the next, the cache
                will be invalidated.
            ignore (iterable of str or None): Arguments to ignore when computing the
                argument hash. This means that changes to those arguments will not cause
                the cache to become invalidated.
            save_hashes_only (bool): If True, only save hash values of the outputs
                instead of the outputs themselves. When this cached function is
                called, outputs will only be recomputed if needed - otherwise the
                saved hash values can be used to retrieve the output of other cached
                functions that depend on the output of this cached function. When
                using this option, additional care has to be taken to avoid mutable
                arguments (e.g. NumPy random Generators, lists, etc...). This is
                useful for functions which take an intermediate amount of time to run
                and are part of a chain of cached functions where the results are only
                needed as intermediaries to arrive at later results.

        Returns:
            callable: The cached function with added `check_in_store()` method.

        """
        if func is None:
            return partial(
                self.cache,
                dependencies=dependencies,
                ignore=ignore,
                save_hashes_only=save_hashes_only,
            )

        assert callable(func)

        # Update dependencies to enable chaining of dependencies.
        dependencies = (
            *dependencies,
            *reduce(
                add,
                map(
                    tuple,
                    map(
                        default_attrgetter("_orig_func._dependencies", default=()),
                        dependencies,
                    ),
                ),
                (),
            ),
        )

        def _inner_placeholder(hashed, args, kwargs):
            if save_hashes_only:
                return func(*args, **kwargs), HASHES_ONLY
            return func(*args, **kwargs)

        _inner_placeholder.__name__ = func.__name__

        cached_inner = self.memory.cache(ignore=["args", "kwargs"])(_inner_placeholder)

        def bound_get_hashed(*orig_args, **orig_kwargs):
            return _get_hashed(
                func,
                *orig_args,
                dependencies=dependencies,
                hash_func=self.get_hash,
                ignore=ignore,
                **orig_kwargs,
            )

        @wraps(func)
        def cached_func(*orig_args, **orig_kwargs):
            hashed = bound_get_hashed(*orig_args, **orig_kwargs)

            if save_hashes_only:
                if cached_inner.store_backend.contains_item(
                    cached_inner._get_output_identifiers(hashed, orig_args, orig_kwargs)
                ):
                    # Do not use the original factory functions since these will reference
                    # data that has never been saved. Only extract the saved hash values.
                    cache_proxies = cached_inner(hashed, orig_args, orig_kwargs)
                    if isinstance(cache_proxies, HashProxy):
                        cached_hash_values = (cache_proxies.hashed_value,)
                    else:
                        cached_hash_values = tuple(
                            proxy.hashed_value for proxy in cache_proxies
                        )
                    # Return a lazy proxy that contains the cached hash values along with
                    # lazy references to the output of the cached function (the
                    # HASHES_ONLY return value is ignored by the backend).

                    def process_func():
                        if hasattr(process_func, "stored"):
                            logger.debug("Returning previously computed data.")
                            return process_func.stored

                        logger.debug("Processing data.")

                        # Call the uncached function here since we have already cached the
                        # hash values. Ignore the additional HASHES_ONLY return value.
                        process_func.stored = _inner_placeholder(
                            hashed, orig_args, orig_kwargs
                        )[0]
                        return process_func.stored

                    if len(cached_hash_values) == 1:
                        return HashProxy(
                            Factory(process_func),
                            hash_func=self.get_hash,
                            hash_value=cached_hash_values[0],
                        )

                    # Otherwise create a lazy proxy for each individual object to associate each
                    # stored object with its individual hash value.

                    def get_factory_func(i):
                        def factory_func():
                            return process_func()[i]

                        return factory_func

                    return tuple(
                        HashProxy(
                            Factory(get_factory_func(i)),
                            hash_func=self.get_hash,
                            hash_value=hash_value,
                        )
                        for i, hash_value in enumerate(cached_hash_values)
                    )

                # If this is the first time the function is called, call it normally and
                # ignore the additional HASHES_ONLY return value.

                return cached_inner(hashed, orig_args, orig_kwargs)[0]

            return cached_inner(hashed, orig_args, orig_kwargs)

        def check_in_store(*args, **kwargs):
            """Check whether a given call to the cached function is already cached.

            Args:
                args, kwargs: Arguments to check.

            Returns:
                IN_STORE: If the given call was found in the cache.

            Raises:
                NotCachedError: If the given call was not found in the cache.

            """
            output_ids = cached_inner._get_output_identifiers(
                bound_get_hashed(*args, **kwargs), args, kwargs
            )
            if not cached_inner.store_backend.contains_item(output_ids):
                raise NotCachedError(f"The given call is not cached: {output_ids}")
            return IN_STORE

        cached_func.check_in_store = check_in_store
        cached_func._orig_func = func
        cached_func._orig_func._dependencies = dependencies

        return cached_func


def get_memory(cache_dir="", **kwargs):
    """Get a joblib Memory object used to cache function results.

    Args:
        cache_dir (str or None): Joblib cache directory name within
            `wildfires.data.DATA_DIR`. If None, no caching will be done.
        **kwargs: Extra arguments passed to `joblib.Memory()`.

    Returns:
        joblib.memory.Memory: Joblib Memory object.

    """
    return Memory(
        location=os.path.join(DATA_DIR, "joblib_cache", cache_dir)
        if cache_dir is not None and data_is_available()
        else None,
        **kwargs,
    )


def checkattr(name):
    """Check if given attributes exist.

    Allows use of dot notation (e.g. name='a.b') due to the use of `operator.attrgetter`

    """

    def check(obj):
        try:
            attrgetter(name)(obj)
            return True
        except AttributeError:
            return False

    return check


def default_attrgetter(*args, default=None):
    """`operator.attrgetter` with a default value."""

    def retrieve(obj):
        try:
            return attrgetter(*args)(obj)
        except AttributeError:
            return default

    return retrieve


def _calculate_dependency_hash(dependencies):
    """Calculate a hash for the dependencies."""
    dependency_hashes = []

    for f in dependencies:
        for flag_check, retrieve_func in (
            # TODO: Make this more robust than simply relying on this ordering.
            # The ordering here is very important since functools.wraps() will
            # copy the '_dependency' flag and so we have to start looking at the
            # deepest possible nesting level.
            (
                checkattr("_orig_func._dependency"),
                attrgetter("_orig_func"),
            ),
            (checkattr("_dependency"), lambda f: f),
        ):
            if flag_check(f):
                func = retrieve_func(f)
                break
        else:
            raise ValueError("All dependencies must be marked with '_dependency'.")

        # Ensure that the hash can be calculated, i.e. that there are no mutable
        # objects present in the default arguments. Copy the object since some
        # object (e.g. immutabledict) will cache the hash resulting from calling
        # `hash()` (e.g. in a '_hash' attribute), and since the output of Python's
        # `hash()` function is not constant across sessions, this causes Joblib's
        # hash to change as well (which we do not want).
        hash(deepcopy(signature(func)))
        # Finally, calculate the hash using Joblib because the inbuilt hash()
        # function changes its output in between runs.
        dependency_hashes.append(joblib.hashing.hash(signature(func)))
        dependency_hashes.append(joblib.hashing.hash(CodeObj(func.__code__).hashable()))

    return joblib.hashing.hash(dependency_hashes)


def mark_dependency(f):
    """Decorator which marks a function as a potential dependency.

    Args:
        f (callable): The dependency to be recorded.

    """
    f._dependency = True
    return f


def _get_hashed(func, *args, dependencies=(), hash_func, ignore=None, **kwargs):
    """Calculate a hash for the call, including dependencies."""
    args, kwargs = extract_uniform_args_kwargs(func, *args, ignore=ignore, **kwargs)

    # Go through the original arguments and hash the contents manually.
    args_hashes = []
    for arg in args:
        args_hashes.append(hash_func(arg))

    # Repeat the above process for the kwargs. The keys should never include
    # MaskedArray data so we only need to deal with the values.
    kwargs_hashes = {}
    for key, arg in kwargs.items():
        kwargs_hashes[key] = hash_func(arg)

    # Hash the original function to differentiate different functions apart.
    func_code = CodeObj(func.__code__).hashable()

    return dict(
        func_code=func_code,
        args_hashes=args_hashes,
        kwargs_hashes=kwargs_hashes,
        dependencies=_calculate_dependency_hash(dependencies),
    )
