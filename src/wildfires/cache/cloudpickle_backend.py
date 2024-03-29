# -*- coding: utf-8 -*-
import datetime
import logging
import os
import re

import cloudpickle
from joblib import register_store_backend
from joblib._store_backends import CacheItemInfo, concurrency_safe_rename, mkdirp

from .backend import CommonBackend
from .hashing import (
    _default_context_managers,
    _default_guarded_hashers,
    _default_initial_hashers,
)

logger = logging.getLogger(__name__)


def register_backend():
    """Register cloudpickle save backend for use with joblib memory."""
    register_store_backend("cloudpickle", CloudpickleStoreBackend)


class CloudpickleStoreBackend(CommonBackend):
    """A StoreBackend used with local or network file systems."""

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    # Run outside of context managers.
    initial_hashers = list(_default_initial_hashers)
    # Context managers that temporarily change objects to enable consistent hashing.
    context_managers = list(_default_context_managers)
    # Run within context managers.
    guarded_hashers = list(_default_guarded_hashers)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match("[a-f0-9]{32}", os.path.basename(dirpath))

            if is_cache_hash_dir:
                output_filename = os.path.join(dirpath, "output.cpkl")
                try:
                    last_access = os.path.getatime(output_filename)
                except OSError:
                    try:
                        last_access = os.path.getatime(dirpath)
                    except OSError:
                        # The directory has already been deleted
                        continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are !!!!
        """
        # TODO: Options for ris netcdf saver in docstring above!

        if backend_options is None:
            backend_options = {}

        # setup location directory
        self.location = location
        if not os.path.exists(self.location):
            mkdirp(self.location)

        self.verbose = verbose

    def load_item(self, path, verbose=1, msg=None):
        """Load an item from the store given its path as a list of strings."""
        full_path = os.path.join(self.location, *path)

        if verbose > 1:
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        filename = os.path.join(full_path, "output.cpkl")
        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        logger.debug(f"cloudpickle loading filename '{filename}'.")
        with open(filename, "rb") as f:
            item = cloudpickle.load(f)
        return item

    def dump_item(self, path, item, verbose=1):
        """Dump an item in the store at the path given as a list of
        strings."""
        try:
            item_path = os.path.join(self.location, *path)
            if not self._item_exists(item_path):
                self.create_location(item_path)
            filename = os.path.join(item_path, "output.cpkl")
            logger.debug(f"Caching '{item}' in '{item_path}'.")
            if verbose > 10:
                print("Persisting in %s" % item_path)

            def write_func(to_write, dest_filename):
                # TODO: Pass options!!
                with open(dest_filename, "wb") as f:
                    cloudpickle.dump(to_write, f, protocol=-1)

            self._concurrency_safe_write(item, filename, write_func)
        except:  # noqa: E722
            " Race condition in the creation of the directory "

    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of strings"""
        item_path = os.path.join(self.location, *path)
        filename = os.path.join(item_path, "output.cpkl")

        return self._item_exists(filename)
