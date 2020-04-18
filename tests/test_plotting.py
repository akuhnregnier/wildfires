#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wildfires.analysis.plotting import FigureSaver, cube_plotting

logging.getLogger("wildfires").setLevel(logging.DEBUG)


@pytest.mark.parametrize("debug,expected_filetype", [(False, ".pdf"), (True, ".png")])
def test_cube_plotting(debug, expected_filetype):
    """Test that `cube_plotting` produces a figure without errors."""
    FigureSaver.debug = debug
    with TemporaryDirectory("plotting") as directory:
        with FigureSaver(directory=directory, filename="cube_test"):
            cube_plotting(np.random.random((100, 100)))
        plt.close()

        assert os.listdir(directory) == [f"cube_test{expected_filetype}"]


@pytest.mark.parametrize("debug,expected_filetype", [(False, ".pdf"), (True, ".png")])
def test_explicit_directory_saving(debug, expected_filetype):
    """Non-debug and debug file types are saved in an explicitly-specified directory."""
    figs = []
    FigureSaver.debug = debug

    with TemporaryDirectory("plotting") as directory:
        with FigureSaver(directories=directory, filenames=("test1", "test2")):
            figs.append(plt.figure())
            figs.append(plt.figure())

        assert set(os.listdir(directory)) == {
            f"test1{expected_filetype}",
            f"test2{expected_filetype}",
        }

    for fig in figs:
        plt.close(fig)


@pytest.mark.parametrize("debug,expected_filetype", [(False, ".pdf"), (True, ".png")])
def test_default_directory_saving(debug, expected_filetype):
    """Test that overriding `FigureSaver.directory` affects the default directory."""
    figs = []
    with TemporaryDirectory("plotting") as directory:
        FigureSaver.directory = directory
        FigureSaver.debug = debug
        with FigureSaver("test1"):
            figs.append(plt.figure())
        with FigureSaver("test2"):
            figs.append(plt.figure())

        assert set(os.listdir(directory)) == {
            f"test1{expected_filetype}",
            f"test2{expected_filetype}",
        }

    for fig in figs:
        plt.close(fig)


@pytest.mark.parametrize("debug,expected_filetype", [(False, ".pdf"), (True, ".png")])
def test_figure_saver_instance(debug, expected_filetype):
    """An FigureSaver instance should be used to set defaults, not the class.

    This is to avoid unintentionally overriding defaults for different modules.

    """

    def get_defaults():
        return [
            getattr(FigureSaver, attr)
            for attr in ("debug", "directory", "options", "debug_options")
        ]

    initial_defaults = get_defaults()
    figs = []
    with TemporaryDirectory("plotting") as directory:
        # Set defaults without affecting the `FigureSaver` defaults.
        figure_saver = FigureSaver(directories=directory, debug=debug)
        # Use this instance with its new defaults (if given) to save figures.
        # In order to enable this functionality, we need to have an object that
        # defines __call__.
        with figure_saver("test1"):
            figs.append(plt.figure())
        with figure_saver("test2"):
            figs.append(plt.figure())

        assert set(os.listdir(directory)) == {
            f"test1{expected_filetype}",
            f"test2{expected_filetype}",
        }

    for fig in figs:
        plt.close(fig)

    final_defaults = get_defaults()

    assert initial_defaults == final_defaults, (
        "By using instances to set defaults, the class defaults should have been left "
        "unchanged."
    )


def test_dpi_saving():
    figs = []
    with TemporaryDirectory("plotting") as directory:
        with FigureSaver(
            "test1", directories=directory, debug=True, filetype=".png", dpi=100
        ):
            figs.append(plt.figure())
            plt.plot([1, 2, 3])
        # Also test multiple ways of using the context manager.
        with FigureSaver(directories=directory, debug=True, filetype=".png", dpi=200)(
            "test2"
        ):
            figs.append(plt.figure())
            plt.plot([1, 2, 3])

        assert os.path.getsize(os.path.join(directory, "test2.png")) > os.path.getsize(
            os.path.join(directory, "test1.png")
        ), "The figure saved at a higher dpi value should be larger."

    for fig in figs:
        plt.close(fig)


def test_directory_creation():
    """New directories should be created if they do not exist.

    This does not apply when the `FigureSaver` default directory is altered
    without instance creation.

    """
    figs = []
    with TemporaryDirectory("plotting") as directory:
        with FigureSaver(
            "test1", directories=os.path.join(directory, "test"), debug=False
        ):
            figs.append(plt.figure())
        with FigureSaver(directories=os.path.join(directory, "test"), debug=False)(
            "test2"
        ):
            figs.append(plt.figure())

        assert set(os.listdir(os.path.join(directory, "test"))) == {
            f"test1.pdf",
            f"test2.pdf",
        }
    for fig in figs:
        plt.close(fig)
