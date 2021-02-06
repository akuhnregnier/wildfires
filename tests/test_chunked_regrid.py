# -*- coding: utf-8 -*-
from functools import reduce
from operator import mul

import dask
import dask.array as da
import iris
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from wildfires.chunked_regrid.core import (
    calculate_blocks,
    convert_chunk_size,
    get_cell_numbers,
    get_overlapping,
    get_valid_cell_mapping,
    join_slices,
    spatial_chunked_regrid,
)

from .utils import simple_cube


@pytest.mark.parametrize(
    "slices, expected",
    [
        ((slice(0, 10),), slice(0, 10)),
        ((slice(0, 10), slice(0, 10)), slice(0, 10)),
        ((slice(0, 10, 1), slice(-2, 0, None)), slice(-2, 10)),
        ((slice(0, 10), slice(-2, 0), slice(10, 20)), slice(-2, 20)),
    ],
)
def test_join_slices(slices, expected):
    assert join_slices(*slices) == expected


def test_join_slices_step():
    with pytest.raises(
        ValueError, match="Slices may not define a step other than 1 or None."
    ):
        join_slices(slice(0, 1, 2))


def test_join_slices_increasing():
    with pytest.raises(ValueError, match="Slices must be increasing."):
        join_slices(slice(2, 1))


def test_join_slices_contiguous():
    with pytest.raises(ValueError, match="All slices must be contiguous."):
        join_slices(slice(1, 2), slice(3, 4))


@pytest.mark.parametrize(
    "source_bounds, target_bounds, exp_inds, exp_src_slice, exp_tgt_slice",
    [
        (
            np.linspace(0, 10, 11),
            np.linspace(2.5, 10.1, 5),
            [[2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9]],
            slice(2, 10),
            slice(0, 4),
        ),
        (
            np.linspace(0, 10, 11),
            np.linspace(0, 10, 5),
            [[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]],
            slice(0, 10),
            slice(0, 4),
        ),
        (
            np.linspace(0, 10, 5),
            np.linspace(0, 10, 11),
            [[0], [0], [0, 1], [1], [1], [2], [2], [2, 3], [3], [3]],
            slice(0, 4),
            slice(0, 10),
        ),
        (
            np.linspace(4, 10, 5),
            np.linspace(0, 10, 5),
            [[], [0], [0, 1, 2], [2, 3]],
            slice(0, 4),
            slice(1, 4),
        ),
        (
            np.linspace(3, 7, 5),
            np.linspace(0, 10, 5),
            [[], [0, 1], [2, 3], []],
            slice(0, 4),
            slice(1, 3),
        ),
        (
            np.linspace(3, 7, 3),
            np.linspace(0, 10, 10),
            [[], [], [0], [0], [0, 1], [1], [1], [], []],
            slice(0, 2),
            slice(2, 7),
        ),
        (
            np.linspace(3, 11, 10),
            np.linspace(0, 10, 10),
            [[], [], [0], [0, 1], [1, 2], [2, 3, 4], [4, 5], [5, 6], [6, 7]],
            slice(0, 8),
            slice(2, 9),
        ),
        (
            np.linspace(-3, 8, 6),
            np.linspace(0, 10, 5),
            [[1, 2], [2, 3], [3, 4], [4]],
            slice(1, 5),
            slice(0, 4),
        ),
        (
            np.linspace(-3, 8, 6),
            np.linspace(0, 11, 5),
            [[1, 2], [2, 3], [3, 4], []],
            slice(1, 5),
            slice(0, 3),
        ),
    ],
)
def test_get_overlapping(
    source_bounds, target_bounds, exp_inds, exp_src_slice, exp_tgt_slice
):
    assert get_overlapping(source_bounds, target_bounds) == (
        exp_inds,
        exp_src_slice,
        exp_tgt_slice,
    )


@pytest.mark.parametrize(
    "source_bounds, target_bounds",
    [
        (
            np.linspace(0, 2, 2),
            np.linspace(-1, 0, 2),
        ),
        (
            np.linspace(1, 2, 2),
            np.linspace(-1, 0, 2),
        ),
        (
            np.linspace(0, 1, 2),
            np.linspace(1, 2, 2),
        ),
    ],
)
def test_get_overlapping_exceptions(source_bounds, target_bounds):
    with pytest.raises(ValueError):
        get_overlapping(source_bounds, target_bounds)


@pytest.mark.parametrize(
    "source_bounds, target_bounds, tolerance, expected",
    [
        (
            np.linspace(0, 1, 4),
            np.linspace(0, 1, 4) + 0.1 - 1e-10,
            0.1,
            [[0], [1], [2]],
        ),
        (
            np.linspace(0, 1, 4),
            np.linspace(0, 1, 4) + 1e-8 - 1e-10,
            1e-8,
            [[0], [1], [2]],
        ),
        (
            [0, 0.25, 0.5],
            [0.25 - 0.99e-6, 0.5, 1],
            1e-6,
            [[1], []],
        ),
        (
            [0, 0.25, 0.5],
            [-1, 0.5 - 0.99e-6, 1],
            1e-6,
            [[0, 1], []],
        ),
    ],
)
def test_get_overlapping_tolerance(source_bounds, target_bounds, tolerance, expected):
    assert get_overlapping(source_bounds, target_bounds, tol=tolerance)[0] == expected


@pytest.mark.parametrize(
    "contained, expected_cell_numbers, expected_overlap",
    [
        (
            [[2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9]],
            [3, 2, 2, 1],
            True,
        ),
        (
            [[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]],
            [3, 2, 3, 2],
            True,
        ),
        (
            [[0], [0], [0, 1], [1], [1], [2], [2], [2, 3], [3], [3]],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            True,
        ),
        (
            [[], [0], [0, 1, 2], [2, 3]],
            [None, 1, 2, 1],
            True,
        ),
        (
            [[], [0, 1], [2, 3], []],
            [None, 2, 2, None],
            False,
        ),
        (
            [[], [], [0], [0], [0, 1], [1], [1], [], []],
            [None, None, 1, 0, 1, 0, 0, None, None],
            True,
        ),
        (
            [[], [], [0], [0, 1], [1, 2], [2, 3, 4], [4, 5], [5, 6], [6, 7]],
            [None, None, 1, 1, 1, 2, 1, 1, 1],
            True,
        ),
        (
            [[1, 2], [2, 3], [3, 4], [4]],
            [2, 1, 1, 0],
            True,
        ),
        (
            [[1, 2], [2, 3], [3, 4], []],
            [2, 1, 1, None],
            True,
        ),
    ],
)
def test_get_cell_numbers(contained, expected_cell_numbers, expected_overlap):
    assert get_cell_numbers(contained) == (expected_cell_numbers, expected_overlap)


@pytest.mark.parametrize(
    "valid_cell_numbers, expected_mapping",
    [
        (
            [3, 2, 2, 1],
            {
                (0,): (0, 1, 2),
                (1,): (3, 4),
                (2,): (5, 6),
                (3,): (7,),
            },
        ),
        (
            [3, 2, 3, 2],
            {
                (0,): (0, 1, 2),
                (1,): (3, 4),
                (2,): (5, 6, 7),
                (3,): (8, 9),
            },
        ),
        (
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            {
                (0, 1): (0,),
                (2, 3, 4): (1,),
                (5, 6): (2,),
                (7, 8, 9): (3,),
            },
        ),
        (
            [1, 2, 1],
            {
                (0,): (0,),
                (1,): (1, 2),
                (2,): (3,),
            },
        ),
        (
            [2, 2],
            {
                (0,): (0, 1),
                (1,): (2, 3),
            },
        ),
        (
            [1, 0, 1, 0, 0],
            {
                (0, 1): (0,),
                (2, 3, 4): (1,),
            },
        ),
        (
            [1, 1, 1, 2, 1, 1, 1],
            {
                (0,): (0,),
                (1,): (1,),
                (2,): (2,),
                (3,): (3, 4),
                (4,): (5,),
                (5,): (6,),
                (6,): (7,),
            },
        ),
        (
            [2, 1, 1, 0],
            {(0,): (0, 1), (1,): (2,), (2, 3): (3,)},
        ),
        (
            [2, 1, 1],
            {
                (0,): (0, 1),
                (1,): (2,),
                (2,): (3,),
            },
        ),
    ],
)
def test_get_valid_cell_mapping(valid_cell_numbers, expected_mapping):
    assert get_valid_cell_mapping(valid_cell_numbers) == expected_mapping


id_mask_funcs = {
    # Simply return False.
    "False": lambda x: False,
    # Mask 1/3 of all elements across the array.
    "3": lambda x: (x.astype("int") % 3) == 0,
    # Block mask.
    "Block": lambda x: x < np.mean(x),
}

id_schemes = {
    "AreaWeighted_0_1": iris.analysis.AreaWeighted(mdtol=0.1),
    "AreaWeighted_1": iris.analysis.AreaWeighted(mdtol=1),
    "Linear": iris.analysis.Linear(extrapolation_mode="mask"),
}

src_tgt_lat_lims = [
    [(-90, 90), (-90, 90)],
    [(-60, 60), (-90, 30)],
    [(-60, 60), (-30, 90)],
]

src_tgt_lon_lims = [
    [(-180, 180), (-180, 180)],
    [(-120, 60), (-180, 0)],
    [(60, 120), (30, 180)],
]

shape_combinations = [
    ((4, 4), (6, 3), ()),
    ((18, 18), (19, 19), ()),
    ((20, 20), (19, 19), ()),
    ((20, 10), (40, 30), ()),
    ((20, 30), (10, 40), ()),
    ((20, 30), (50, 10), ()),
    ((100, 200), (50, 120), pytest.mark.slow),
    ((200, 300), (150, 201), pytest.mark.slow),
]


@pytest.mark.parametrize("max_chunk_size", (None, 20, "20KB", "1MB"))
@pytest.mark.parametrize(
    "mask_func",
    id_mask_funcs.values(),
    ids=id_mask_funcs.keys(),
)
@pytest.mark.parametrize("scheme", id_schemes.values(), ids=id_schemes.keys())
@pytest.mark.parametrize(
    "src_lon_lims, tgt_lon_lims",
    src_tgt_lon_lims,
    ids=[
        "-".join(f"{'_'.join(map(str, lims))}" for lims in lon_lims)
        for lon_lims in src_tgt_lon_lims
    ],
)
@pytest.mark.parametrize(
    "src_lat_lims, tgt_lat_lims",
    src_tgt_lat_lims,
    ids=[
        "-".join(f"{'_'.join(map(str, lims))}" for lims in lat_lims)
        for lat_lims in src_tgt_lat_lims
    ],
)
@pytest.mark.parametrize(
    "src_shape, tgt_shape",
    [
        pytest.param(
            src_shape,
            tgt_shape,
            marks=marks,
            id="-".join(
                f"{'_'.join(map(str, shape))}" for shape in (src_shape, tgt_shape)
            ),
        )
        for (src_shape, tgt_shape, marks) in shape_combinations
    ],
)
def test_regrid(
    src_shape,
    tgt_shape,
    src_lat_lims,
    tgt_lat_lims,
    src_lon_lims,
    tgt_lon_lims,
    scheme,
    mask_func,
    max_chunk_size,
):
    src_data = da.arange(reduce(mul, src_shape)).reshape(src_shape)
    src_cube = simple_cube(
        da.ma.masked_array(
            src_data,
            mask=mask_func(src_data),
        ),
        lat_lims=src_lat_lims,
        lon_lims=src_lon_lims,
        long_name="src",
    )
    tgt_cube = simple_cube(
        da.ma.masked_array(
            da.zeros(tgt_shape),
            mask=False,
        ),
        lat_lims=tgt_lat_lims,
        lon_lims=tgt_lon_lims,
        long_name="tgt",
    )

    exp_reg_cube = src_cube.copy().regrid(tgt_cube, scheme)

    # Use the local threaded scheduler for testing.
    with dask.config.set(scheduler="single-threaded"):
        new_reg_cube = spatial_chunked_regrid(
            src_cube,
            tgt_cube,
            scheme,
            max_src_chunk_size=max_chunk_size,
            max_tgt_chunk_size=max_chunk_size,
        )
    assert new_reg_cube.has_lazy_data()
    assert isinstance(new_reg_cube.core_data()._meta, np.ma.MaskedArray)
    assert_array_equal(exp_reg_cube.data.mask, new_reg_cube.data.mask)
    mask = exp_reg_cube.data.mask
    assert_allclose(exp_reg_cube.data.data[~mask], new_reg_cube.data.data[~mask])


@pytest.mark.parametrize(
    "chunk_size, factor, dtype, masked, expected",
    [
        ("100B", 1, np.float32(), False, 25),
        ("1000B", 1, np.float32(), False, 250),
        ("1KB", 1, np.float32(), False, 250),
        ("1KB", 1, np.float64(), False, 125),
        ("1KB", 1, np.float64(), True, 111),
        ("1KB", 10, np.float64(), True, 11),
        ((None,) * 5),
        ((1,) * 5),
    ],
)
def test_convert_chunk_size(chunk_size, factor, dtype, masked, expected):
    assert convert_chunk_size(chunk_size, factor, dtype, masked) == expected


@pytest.mark.parametrize(
    "src_chunks, tgt_chunks, tgt_slices, min_src_chunk_size, max_src_chunk_size,"
    "min_tgt_chunk_size, max_tgt_chunk_size, expected_src_chunks, expected_tgt_slices",
    [
        (
            # Input data.
            [1, 1],
            [1, 1],
            [slice(0, 1), slice(1, 2)],
            # Constraints.
            None,
            None,
            None,
            None,
            # Expected output.
            (2,),
            (slice(0, 2),),
        ),
        (
            # Input data.
            [1, 2, 1, 3],
            [1, 1, 2, 1],
            (slice(0, 1), slice(1, 2), slice(2, 4), slice(4, 5)),
            # Constraints.
            None,
            3,
            None,
            None,
            # Expected output.
            (3, 1, 3),
            (slice(0, 2), slice(2, 4), slice(4, 5)),
        ),
    ],
)
def test_calculate_blocks(
    src_chunks,
    tgt_chunks,
    tgt_slices,
    min_src_chunk_size,
    max_src_chunk_size,
    min_tgt_chunk_size,
    max_tgt_chunk_size,
    expected_src_chunks,
    expected_tgt_slices,
):
    """Test the combination of source and target chunks given constraints."""
    assert (
        calculate_blocks(
            src_chunks,
            tgt_chunks,
            tgt_slices,
            min_src_chunk_size,
            max_src_chunk_size,
            min_tgt_chunk_size,
            max_tgt_chunk_size,
        )
        == (expected_src_chunks, expected_tgt_slices)
    )


def test_calculate_blocks_max_src_exception():
    with pytest.raises(
        ValueError, match="Original source chunks cannot exceed the limit."
    ):
        calculate_blocks([10], [5], [slice(0, 5)], 1, 5, 1, 10)


def test_calculate_blocks_max_tgt_exception():
    with pytest.raises(
        ValueError, match="Original target chunks cannot exceed the limit."
    ):
        calculate_blocks([10], [5], [slice(0, 5)], 1, 20, 1, 2)


def test_calculate_blocks_min_max_src_exception():
    with pytest.raises(
        ValueError,
        match="Minimum source chunk size cannot exceed the maximum source chunk size",
    ):
        calculate_blocks([10], [5], [slice(0, 5)], 2, 1, None, None)


def test_calculate_blocks_min_max_tgt_exception():
    with pytest.raises(
        ValueError,
        match="Minimum target chunk size cannot exceed the maximum target chunk size",
    ):
        calculate_blocks([10], [5], [slice(0, 5)], None, None, 2, 1)
