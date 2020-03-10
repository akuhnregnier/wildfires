#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from wildfires.analysis.plotting import FigureSaver, cube_plotting

logging.getLogger("wildfires").setLevel(logging.DEBUG)


def test_figure_saving():
    with TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as directory:
        with FigureSaver(directory=directory, filename="test"):
            plt.figure()

        assert os.listdir(directory) == ["test.pdf"]

    FigureSaver.debug = True

    with TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as directory:
        with FigureSaver(directory=directory, filename=("test1", "test2")):
            plt.figure()
            plt.figure()

        assert set(os.listdir(directory)) == {"test1.png", "test2.png"}

    with TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as directory:
        with FigureSaver(directory=directory, filename="cube_test"):
            cube_plotting(np.random.random((100, 100)))

        assert os.listdir(directory) == ["cube_test.png"]
