# -*- coding: utf-8 -*-
import dask.array as da
import iris
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from wildfires.data.landcover import (
    conversion,
    convert_to_pfts,
    get_mapping_arrays,
    get_mapping_pfts,
)


def test_categories():
    """Ensure the correct number of land cover categories are present."""
    # Total number of land cover categories.
    assert len(conversion) == 37


def test_ds():
    """Ensure the correct number of PFTs are present."""
    # Extract all PFTs.
    pfts = set()
    for lc_data in conversion.values():
        pfts.update(lc_data["pfts"])
    assert pfts == {
        "Tree.BE",
        "Tree.BD",
        "Tree.NE",
        "Tree.ND",
        "Shrub.BE",
        "Shrub.BD",
        "Shrub.NE",
        "Herb",
        "Crop",
        "Bare",
        "NoLand",
    }


def test_100_coverage():
    """Ensure all categories have PFT coverages adding up to 100%."""
    for lc_id, lc_data in conversion.items():
        assert (
            sum(lc_data["pfts"].values()) == 100
        ), f"PFTs for '{lc_id}: {lc_data['name']}' do not add up to 100%."


@pytest.fixture
def fixture_mapping():
    """Mapping from dummy categories to PFT fractions."""
    return ("a", "b", "c"), {
        0: {
            "pfts": {
                "a": 50,
                "b": 50,
            },
        },
        1: {
            "pfts": {
                "a": 20,
                "b": 50,
                "c": 30,
            },
        },
        3: {
            "pfts": {
                "c": 100,
            },
        },
    }


def test_get_mapping_pfts(fixture_mapping):
    assert_array_equal(get_mapping_pfts(fixture_mapping[1]), fixture_mapping[0])


def test_get_mapping_arrays(fixture_mapping):
    conv_mapping = get_mapping_arrays(*fixture_mapping)
    assert all(
        isinstance(values["pfts"], np.ndarray) for values in conv_mapping.values()
    )
    assert_array_equal(conv_mapping[0]["pfts"], [50, 50, 0])
    assert_array_equal(conv_mapping[1]["pfts"], [20, 50, 30])
    assert_array_equal(conv_mapping[3]["pfts"], [0, 0, 100])
    for index in (0, 1, 3):
        assert conv_mapping[index]["pfts"].dtype == np.uint8


@pytest.mark.parametrize("chunks", ("auto", -1, 2))
def test_conversion(fixture_mapping, chunks):
    cats = iris.cube.Cube(
        da.from_array(
            np.array(
                [[[1, 0, 0], [0, 0, 2], [3, 2, 2], [0, 3, 1], [3, 3, 0], [0, 1, 1]]],
                dtype=np.uint8,
            ),
            chunks=chunks,
        )
    )
    pfts = convert_to_pfts(cats, fixture_mapping[1], 0, 3)

    # NOTE: This check done on the non-lazy '.data' attribute fails, as Iris seems to
    # implicitly convert everything to np.float64.
    for i in range(3):
        assert pfts[i].core_data().dtype == np.uint8

    # PFT: a.
    assert_array_equal(
        pfts[0].data,
        np.array(
            [
                [
                    [20, 50, 50],
                    [50, 50, 0],
                    [0, 0, 0],
                    [50, 0, 20],
                    [0, 0, 50],
                    [50, 20, 20],
                ]
            ]
        ),
    )
    # PFT: b.
    assert_array_equal(
        pfts[1].data,
        np.array(
            [
                [
                    [50, 50, 50],
                    [50, 50, 0],
                    [0, 0, 0],
                    [50, 0, 50],
                    [0, 0, 50],
                    [50, 50, 50],
                ]
            ]
        ),
    )
    # PFT: c.
    assert_array_equal(
        pfts[2].data,
        np.array(
            [
                [
                    [30, 0, 0],
                    [0, 0, 0],
                    [100, 0, 0],
                    [0, 100, 30],
                    [100, 100, 0],
                    [0, 30, 30],
                ]
            ]
        ),
    )
