# -*- coding: utf-8 -*-
import shutil
from contextlib import ExitStack

import joblib
from joblib._store_backends import (
    StoreBackendBase,
    StoreBackendMixin,
    concurrency_safe_write,
    mkdirp,
    rm_subdirs,
)


class HashBackendMixin:
    """Hashing-related functionality."""

    @classmethod
    def get_hash(cls, arg):
        """Compute a hash with special support for e.g. MaskedArray."""
        for hasher in cls.initial_hashers:
            if hasher.test_argument(arg):
                return hasher.hash(arg)
        with ExitStack() as stack:
            for manager in cls.context_managers:
                stack.enter_context(manager(arg))

            for hasher in cls.guarded_hashers:
                if hasher.test_argument(arg):
                    return hasher.hash(arg)

            # Default case.
            return joblib.hashing.hash(arg)


class CommonBackend(StoreBackendBase, StoreBackendMixin, HashBackendMixin):
    """Common storage-related functionality."""

    def clear_location(self, location):
        """Delete location on store."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        temporary_filename = concurrency_safe_write(to_write, filename, write_func)
        self._move_item(temporary_filename, filename)
