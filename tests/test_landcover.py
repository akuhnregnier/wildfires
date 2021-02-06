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
                "a": 0.5,
                "b": 0.5,
            },
        },
        1: {
            "pfts": {
                "a": 0.2,
                "b": 0.5,
                "c": 0.3,
            },
        },
        3: {
            "pfts": {
                "c": 1.0,
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
    assert_array_equal(conv_mapping[0]["pfts"], [0.5, 0.5, 0])
    assert_array_equal(conv_mapping[1]["pfts"], [0.2, 0.5, 0.3])
    assert_array_equal(conv_mapping[3]["pfts"], [0.0, 0.0, 1.0])


@pytest.mark.parametrize("chunks", ("auto", -1, 2))
def test_conversion(fixture_mapping, chunks):
    cats = iris.cube.Cube(
        da.from_array(
            np.array(
                [[[1, 0, 0], [0, 0, 2], [3, 2, 2], [0, 3, 1], [3, 3, 0], [0, 1, 1]]]
            ),
            chunks=chunks,
        )
    )
    pfts = convert_to_pfts(
        cats, fixture_mapping[0], get_mapping_arrays(*fixture_mapping)
    )

    # PFT: a.
    assert_array_equal(
        pfts[0].data,
        np.array(
            [
                [
                    [0.2, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.2],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.2, 0.2],
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
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.5, 0.5],
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
                    [0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.3],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.3, 0.3],
                ]
            ]
        ),
    )
