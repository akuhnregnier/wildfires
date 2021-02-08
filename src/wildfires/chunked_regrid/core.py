# -*- coding: utf-8 -*-
from itertools import islice

import dask
import dask.array as da
import iris
import numpy as np
from dask.utils import parse_bytes
from iris.analysis._interpolation import get_xy_dim_coords

__all__ = (
    "spatial_chunked_regrid",
    "slice_len",
    "join_slices",
)


def get_overlapping(source_bounds, target_bounds, tol=1e-8):
    """Assumes contiguous and monotonically increasing bounds.

    Args:
        source_bounds, target_bounds (1-D array-like): Source and target cell bounds
            along the same dimension.
        tol (float): Absolute floating point tolerance within which boundaries are
            considered equal.

    Returns:
        overlapping (list): List of indices corresponding to the overlapping source cells. One such
            list will be returned for every target cell,
            i.e. length `len(target_bounds) - 1`.
        src_slice, tgt_slice (slice): Valid start and end source/target cell indices.

    """
    contained = [[] for _ in range(len(target_bounds) - 1)]

    target_start_index = 0

    # If the target grid starts before the source grid, trim the target grid
    # before progressing.
    # There are entries to trim if target_start_index is > 0.
    for target_start_index, upper_t in enumerate(target_bounds[1:]):
        if (upper_t - tol) > source_bounds[0]:
            break
    else:
        # If the loop runs until completion, the grids do not overlap.
        raise ValueError("No overlap between the given grids.")

    target_end_index = len(target_bounds)

    # If the target grid ends after the source grid, trim the target grid before
    # progressing.
    # There are entries to trim if target_end_index is < len(target_end_index) - 1.
    for target_end_index, lower_t in zip(
        range(len(target_bounds) - 1, 0, -1), target_bounds[-2::-1]
    ):
        if (lower_t + tol) < source_bounds[-1]:
            break
    else:
        target_end_index -= 1
    if target_end_index == 0:
        raise ValueError("No overlap between the given grids.")

    # Lowest target bound.
    first_lower_t = target_bounds[0]

    # Determine the index corresponding to the first relevant source cell.
    for i, upper_s in enumerate(source_bounds[1:]):
        if (upper_s - tol) > first_lower_t:
            start_i = i
            break

    def target_bounds_iter():
        for (index, (lower_t, upper_t)) in enumerate(
            zip(
                target_bounds[:-1],
                target_bounds[1:],
            )
        ):
            yield index, lower_t, upper_t

    source_start_index = start_i

    for index, lower_t, upper_t in islice(
        target_bounds_iter(), target_start_index, target_end_index
    ):
        for (i, (lower_s, upper_s)) in enumerate(
            zip(
                source_bounds[start_i:-1],
                source_bounds[start_i + 1 :],
            )
        ):
            if np.isclose(upper_s, upper_t, rtol=0, atol=tol):
                contained[index].append(start_i + i)
                start_i += i + 1
                break
            elif (lower_s + tol) < upper_t:
                contained[index].append(start_i + i)
            else:
                start_i += i - 1
                break
        else:
            # We didn't encounter a break while handling the last source region.
            # This is akin to the normal addition of 'i - 1', but since this would
            # normally happen in the next iteration (which cannot happen now since we
            # are handling the last source region) we simply add 'i' here instead.
            start_i += i

    source_end_index = contained[index][-1] + 1
    return (
        contained,
        slice(source_start_index, source_end_index),
        slice(target_start_index, target_end_index),
    )


def get_cell_numbers(contained):
    """Retrieve non-overlapping cell numbers from the output of `get_overlapping`.

    None may appear at the ends of the output, indicating that the corresponding
    target cells are not overlapping with any source cells. These should be ignored
    when regridding.

    Cell numbers of 0 indicate that the corresponding target cells need to be regridded
    in combination with the previous non-zero cell number target cell.

    Returns:
        cell_numbers (list): The number of cells corresponding to the source
            dimension, as described above.
        overlap (bool): If True, this indicates that for at least one location, there
            is an overlap of depth 1 between adjacent operations.

    """
    cell_numbers = []
    overlap = False
    for prev_elements, elements in zip([None] + contained[:-1], contained):
        cell_number = None
        if (
            prev_elements is not None
            and elements
            and prev_elements
            and elements[0] == prev_elements[-1]
        ):
            overlap = True
            cell_number = -1
        if elements:
            if cell_number is None:
                cell_number = 0
            cell_number += elements[-1] - elements[0] + 1

        cell_numbers.append(cell_number)
    return cell_numbers, overlap


def get_valid_cell_mapping(valid_cell_numbers):
    """Calculate which source cells are needed for each target cell.

    This returns a mapping: target cells -> source cells (indices), with source and
    target cell indices starting at 0, i.e. at the first valid target/source cells!

    Raises:
        ValueError: If the first or last valid cell number is None.
        ValueError: If the first valid cell number is 0.

    """
    if valid_cell_numbers[0] is None or valid_cell_numbers[-1] is None:
        raise ValueError("First and last valid cell number must be valid.")
    if valid_cell_numbers[0] == 0:
        raise ValueError("First valid cell number must be non-zero.")

    mapping = {}
    curr_tgt_cells = []
    curr_src_cells = []
    src_cell_count = 0
    for tgt_index, cell_number in enumerate(valid_cell_numbers):
        if cell_number and curr_tgt_cells:
            # Write current state to the mapping and reset.
            mapping[tuple(curr_tgt_cells)] = tuple(curr_src_cells)
            curr_tgt_cells = []
            curr_src_cells = []

        curr_tgt_cells.append(tgt_index)
        for _ in range(cell_number):
            curr_src_cells.append(src_cell_count)
            src_cell_count += 1

    # Finally write the last state to the mapping.
    mapping[tuple(curr_tgt_cells)] = tuple(curr_src_cells)

    return mapping


def regrid_chunk(
    block_src_data,
    block_src_y_pnts,
    block_src_low_y_bnds,
    block_src_upp_y_bnds,
    src_y_coord_metadata,
    src_x_coord,
    src_cube_metadata,
    tgt_y_coord,
    tgt_x_coord,
    tgt_y_slices,
    tgt_cube_metadata,
    y_dim,
    x_dim,
    scheme,
    block_info=None,
):
    # Construct source and target cubes.
    block_src_y_coord = iris.coords.DimCoord(
        block_src_y_pnts.ravel(),
        bounds=np.hstack(
            (
                block_src_low_y_bnds,
                block_src_upp_y_bnds,
            )
        ),
    )
    block_src_y_coord.metadata = src_y_coord_metadata

    src_cube = iris.cube.Cube(
        block_src_data,
        dim_coords_and_dims=[(block_src_y_coord, y_dim), (src_x_coord, x_dim)],
    )
    src_cube.metadata = src_cube_metadata

    tgt_y_slice = tgt_y_slices[block_info[0]["chunk-location"][0]]
    block_tgt_y_coord = tgt_y_coord[tgt_y_slice]

    tgt_shape = (
        block_tgt_y_coord.shape[0],
        tgt_x_coord.shape[0],
    )
    tgt_cube = iris.cube.Cube(
        da.empty(tgt_shape),
        dim_coords_and_dims=[(block_tgt_y_coord, y_dim), (tgt_x_coord, x_dim)],
    )
    tgt_cube.metadata = tgt_cube_metadata

    # Regrid and ensure that there are 2 dimensions.
    reg_data = src_cube.regrid(tgt_cube, scheme).data.reshape(tgt_shape)
    return reg_data


def slice_len(s):
    """Compute the 'length' of a slice, i.e. stop - start.

    Args:
        s (slice): Slice object.

    Raises:
        ValueError: If `s` has a step other than 1 or None.
        ValueError: If `s` is decreasing.

    """
    if s.step not in (None, 1):
        raise ValueError("Slices may not define a step other than 1 or None.")
    if s.stop < s.start:
        raise ValueError("Slice must not be decreasing.")
    return s.stop - s.start


def join_slices(*slices):
    """Join together contiguous slices by extending their start and/or end points.

    All slices must be increasing, i.e. start < stop.
    Identical slices will be removed.

    Args:
        *slices (iterable of slice): Slices to concatenate.

    Examples:
        >>> join_slices(slice(0, 2), slice(2, 4))
        slice(0, 4, None)
        >>> join_slices(slice(0, 2), slice(2, 4), slice(-2, 0))
        slice(-2, 4, None)

    Raises:
        ValueError: If any of the given slices defines a step other than 1 or None.
        ValueError: If the slices are not contiguous.
        ValueError: If any of the slices are not increasing.

    """
    if not slices:
        raise ValueError("No slices were given.")
    for s in slices:
        if s.step not in (None, 1):
            raise ValueError("Slices may not define a step other than 1 or None.")
        if s.stop < s.start:
            raise ValueError("Slices must be increasing.")

    if len(slices) == 1:
        return slices[0]

    # Sort unique slices by their start.
    slices = sorted(
        (
            slice(*elements)
            for elements in set((s.start, s.stop, s.step) for s in slices)
        ),
        key=lambda s: s.start,
    )

    # Ensure they are contiguous.
    for s1, s2 in zip(slices[:-1], slices[1:]):
        if s1.stop != s2.start:
            raise ValueError("All slices must be contiguous.")

    return slice(slices[0].start, slices[-1].stop)


def calculate_blocks(
    src_chunks,
    tgt_chunks,
    tgt_slices,
    min_src_chunk_size,
    max_src_chunk_size,
    min_tgt_chunk_size,
    max_tgt_chunk_size,
):
    """Combine source and target chunks to satisfy the given constraints.

    The maximum number of joins possible will be carried out.

    Args:
        src_chunks, tgt_chunks (iterable of int): Chunk sizes to combine. Note that
            these variables can be though of as a mapping, with each chunk in
            `src_chunks` being required to compute the corresponding chunk in
            `tgt_chunks`. All chunks are contiguous.
        tgt_slices (iterable of slice): Slices to combine in accordance with the
            combinations carried out on `src_chunks` and `tgt_chunks`.
        min_src_chunk_size, min_tgt_chunk_size (None, int): Minimum source (src) and
            target (tgt) chunk sizes.
        max_src_chunk_size, max_tgt_chunk_size (None, int): Maximum source (src) and
            target (tgt) chunk sizes.

    Returns:
        tuple of int: Recalculated source chunks.
        tuple of slice: Recalculated target slices.

    Raises:
        ValueError: If no blocks can be calculated to satisfy the chunk size
            constraints.

    """
    if max_src_chunk_size is None:
        max_src_chunk_size = np.inf
    if max_tgt_chunk_size is None:
        max_tgt_chunk_size = np.inf
    if min_src_chunk_size is None:
        min_src_chunk_size = 1
    if min_tgt_chunk_size is None:
        min_tgt_chunk_size = 1
    min_max_err = "Minimum {0:} chunk size cannot exceed the maximum {0:} chunk size."
    if min_src_chunk_size > max_src_chunk_size:
        raise ValueError(min_max_err.format("source"))
    if min_tgt_chunk_size > max_tgt_chunk_size:
        raise ValueError(min_max_err.format("target"))

    src_chunks = list(src_chunks)
    tgt_chunks = list(tgt_chunks)
    tgt_slices = list(tgt_slices)

    orig_chunk_err = "Original {:} chunks cannot exceed the limit (max: {:} > {:})."
    if any(src_chunk > max_src_chunk_size for src_chunk in src_chunks):
        raise ValueError(
            orig_chunk_err.format("source", max(src_chunks), max_src_chunk_size)
        )
    if any(tgt_chunk > max_tgt_chunk_size for tgt_chunk in tgt_chunks):
        raise ValueError(
            orig_chunk_err.format("target", max(tgt_chunks), max_tgt_chunk_size)
        )

    if len(src_chunks) == 1:
        # Nothing to join.
        return tuple(src_chunks), tuple(tgt_slices)

    index = 0
    while index < (len(src_chunks) - 1):
        increment = 2
        if (
            sum(src_chunks[index : index + increment]) <= max_src_chunk_size
            and sum(tgt_chunks[index : index + increment]) <= max_tgt_chunk_size
        ):
            # Join the two chunks.
            src_chunks = (
                src_chunks[:index]
                + [sum(src_chunks[index : index + increment])]
                + src_chunks[index + increment :]
            )

            tgt_chunks = (
                tgt_chunks[:index]
                + [sum(tgt_chunks[index : index + increment])]
                + tgt_chunks[index + increment :]
            )

            tgt_slices = (
                tgt_slices[:index]
                + [join_slices(*tgt_slices[index : index + increment])]
                + tgt_slices[index + increment :]
            )
            # Try to join the combined item and the next item again.
        else:
            # Move on to the next item.
            index += 1

    return tuple(src_chunks), tuple(tgt_slices)


def convert_chunk_size(chunk_size, factor, dtype, masked):
    """Convert a chunk size given as a string to number of elements.

    Args:
        chunk_size (None, int, str): Chunk size. Conversion using this function is
            only required if a string is given - other values are simply returned.
        factor (int): Number of elements in dimensions which are not chunked, e.g. the
            number of rows if chunking is done exclusively along columns. This is used
            to compute the size per chunk.
        dtype (numpy dtype): Data dtype.
        masked (bool): If True, an additional byte per element will be taken into
            account.

    Returns:
        int or None: Maximum number of elements to stay below the given `chunk_size`.
            If `chunk_size` is None, None will be returned.

    """
    if chunk_size is None or isinstance(chunk_size, int):
        # Do nothing.
        return chunk_size
    element_size = (dtype.itemsize + (1 if masked else 0)) * factor
    return parse_bytes(chunk_size) // element_size


def spatial_chunked_regrid(
    src_cube,
    tgt_cube,
    scheme,
    min_src_chunk_size=2,
    max_src_chunk_size=dask.config.get("array.chunk-size"),
    min_tgt_chunk_size=None,
    max_tgt_chunk_size=dask.config.get("array.chunk-size"),
    tol=1e-16,
):
    """Spatially chunked regridding using dask.

    Only the y-coordinate is chunked. This is done because the x-coordinate may be
    circular (global), which may require additional logic to be implemented.

    Args:
        src_cube (iris.cube.Cube): Cube to be regridded onto the coordinate system
            defined by the target cube.
        tgt_cube (iris.cube.Cube): Target cube. This is solely required to specify the
            target coordinate system and may contain dummy data.
        scheme: The type of regridding to use to regrid the source cube onto the
            target grid, e.g. `iris.analysis.Linear`, `iris.analysis.Nearest`, and
            `iris.analysis.AreaWeighted`.
        min_src_chunk_size (None, int): Minimum source cube chunk size along the
            y-dimension, specified in terms of the number of elements per chunk along
            this axis. Note that some regridders, e.g. `iris.analysis.Linear()`
            require at least a chunk size of 2 here.
        max_src_chunk_size (None, int, str): The maximum size of chunks along the
            source cube's y-dimension. Can be given in bytes, e.g. '10MB' or '100KB'.
            If None is given, the chunks will be as large as possible.
        min_tgt_chunk_size (None, int): Analogous to `min_src_chunk_size` for the
            target cube.
        max_tgt_chunk_size (None, int, str): Analogous to `max_src_chunk_size` for the
            target cube.

    Raises:
        TypeError: If `src_cube` does not have lazy data.
        ValueError: If the source cube is not 2D.
        ValueError: If the source or target cube do not define x and y coordinates.
        ValueError: If source and target cubes do not define their x and y coordinates
            along the same dimensions.
        ValueError: If any of the x, y coordinates are not monotonic.
        ValueError: If the given maximum chunk sizes are smaller than required for the
            regridding of a single data chunk.

    """
    if not src_cube.has_lazy_data():
        raise TypeError("Source cube needs to have lazy data.")
    if src_cube.core_data().ndim != 2:
        raise ValueError("Source cube data needs to be 2D.")

    coord_err = "{name} cube needs to define x and y coordinates."
    try:
        src_x_coord, src_y_coord = get_xy_dim_coords(src_cube)
    except Exception as exc:
        raise ValueError(coord_err.format("Source")) from exc
    try:
        tgt_x_coord, tgt_y_coord = get_xy_dim_coords(tgt_cube)
    except Exception as exc:
        raise ValueError(coord_err.format("Target")) from exc

    y_dim = src_y_dim = src_cube.coord_dims(src_y_coord)[0]
    x_dim = src_x_dim = src_cube.coord_dims(src_x_coord)[0]

    tgt_y_dim = tgt_cube.coord_dims(tgt_y_coord)[0]
    tgt_x_dim = tgt_cube.coord_dims(tgt_x_coord)[0]

    if (src_y_dim, src_x_dim) != (tgt_y_dim, tgt_x_dim):
        raise ValueError("Coordinates are not aligned.")

    monotonic_err_msg = "{:}-coordinate needs to be monotonic."

    src_x_coord_monotonic, src_x_coord_direction = iris.util.monotonic(
        src_x_coord.points, return_direction=True
    )
    if not src_x_coord_monotonic:
        raise ValueError(monotonic_err_msg.format("Source x"))
    if src_x_coord_direction < 0:
        # Coordinate is monotonically decreasing, so we need to invert it.
        flip_slice = [slice(None)] * src_cube.ndim
        flip_slice[src_x_dim] = slice(None, None, -1)
        src_cube = src_cube[tuple(flip_slice)]
        src_x_coord, src_y_coord = get_xy_dim_coords(src_cube)
        src_x_coord.bounds = src_x_coord.bounds[:, ::-1]

    src_y_coord_monotonic, src_y_coord_direction = iris.util.monotonic(
        src_y_coord.points, return_direction=True
    )
    if not src_y_coord_monotonic:
        raise ValueError(monotonic_err_msg.format("Source y"))
    if src_y_coord_direction < 0:
        # Coordinate is monotonically decreasing, so we need to invert it.
        flip_slice = [slice(None)] * src_cube.ndim
        flip_slice[src_y_dim] = slice(None, None, -1)
        src_cube = src_cube[tuple(flip_slice)]
        src_x_coord, src_y_coord = get_xy_dim_coords(src_cube)
        src_y_coord.bounds = src_y_coord.bounds[:, ::-1]

    tgt_x_coord_monotonic, tgt_x_coord_direction = iris.util.monotonic(
        tgt_x_coord.points, return_direction=True
    )
    if not tgt_x_coord_monotonic:
        raise ValueError(monotonic_err_msg.format("Target x"))
    if tgt_x_coord_direction < 0:
        # Coordinate is monotonically decreasing, so we need to invert it.
        flip_slice = [slice(None)] * tgt_cube.ndim
        flip_slice[tgt_x_dim] = slice(None, None, -1)
        tgt_cube = tgt_cube[tuple(flip_slice)]
        tgt_x_coord, tgt_y_coord = get_xy_dim_coords(tgt_cube)
        tgt_x_coord.bounds = tgt_x_coord.bounds[:, ::-1]

    tgt_y_coord_monotonic, tgt_y_coord_direction = iris.util.monotonic(
        tgt_y_coord.points, return_direction=True
    )
    if not tgt_y_coord_monotonic:
        raise ValueError(monotonic_err_msg.format("Target y"))
    if tgt_y_coord_direction < 0:
        # Coordinate is monotonically decreasing, so we need to invert it.
        flip_slice = [slice(None)] * tgt_cube.ndim
        flip_slice[tgt_y_dim] = slice(None, None, -1)
        tgt_cube = tgt_cube[tuple(flip_slice)]
        tgt_x_coord, tgt_y_coord = get_xy_dim_coords(tgt_cube)
        tgt_y_coord.bounds = tgt_y_coord.bounds[:, ::-1]

    max_src_chunk_size = convert_chunk_size(
        max_src_chunk_size,
        # The number of elements along the non-chunked dimension.
        factor=src_cube.shape[x_dim],
        dtype=src_cube.dtype,
        masked=isinstance(src_cube.core_data()._meta, np.ma.MaskedArray),
    )
    max_tgt_chunk_size = convert_chunk_size(
        max_tgt_chunk_size,
        # The number of elements along the non-chunked dimension.
        factor=tgt_cube.shape[x_dim],
        # NOTE: Is this true?
        dtype=src_cube.dtype,
        # Set masked to True here since we will add a mask later in all cases.
        masked=True,
    )
    max_chunk_msg = (
        "Maximum {:} chunk size was smaller than the minimum required for a single "
        "chunk."
    )
    if max_src_chunk_size == 0:
        raise ValueError(max_chunk_msg.format("source"))
    if max_tgt_chunk_size == 0:
        raise ValueError(max_chunk_msg.format("target"))

    # Calculate all possible chunks along the y dimension.
    overlap_indices, valid_src_y_slice, valid_tgt_y_slice = get_overlapping(
        src_y_coord.contiguous_bounds(),
        tgt_y_coord.contiguous_bounds(),
        tol=tol,
    )
    # Some regridding methods care about cells with points outside of overlapping
    # bounds, like `iris.analysis.Linear()`. Include an additional source cell on
    # either end if possible to account for this.
    # NOTE: These additions may be superfluous, but would require re-writing
    # `get_overlapping()` to determine.
    if valid_src_y_slice.start > 0:
        overlap_indices[valid_tgt_y_slice][0].insert(
            0, overlap_indices[valid_tgt_y_slice][0][0] - 1
        )
    if valid_src_y_slice.stop < src_y_coord.shape[0]:
        overlap_indices[valid_tgt_y_slice][-1].append(
            overlap_indices[valid_tgt_y_slice][-1][-1] + 1
        )
    valid_src_y_slice = slice(
        max(0, valid_src_y_slice.start - 1),
        min(src_y_coord.shape[0], valid_src_y_slice.stop + 1),
    )

    cell_numbers, overlap_y = get_cell_numbers(overlap_indices)
    cell_mapping = get_valid_cell_mapping(cell_numbers[valid_tgt_y_slice])

    tgt_y_slices = []
    src_y_chunks = []
    tgt_y_chunks = []

    for tgt_cells, src_cells in cell_mapping.items():
        tgt_y_slices.append(slice(tgt_cells[0], tgt_cells[-1] + 1))
        src_y_chunks.append(len(src_cells))
        tgt_y_chunks.append(len(tgt_cells))

    # XXX: This override is sometimes needed due to floating point errors, e.g. for
    # test_regrid case
    # 100_200-50_120--90_90--90_90--180_180--180_180-AreaWeighted_1-Block-20KB
    # where tgt_cube.coord('latitude')[24].bounds[0][1] is 3.55e-15 instead of 0,
    # causing src_cube.coord('latitude')[50] (with bounds [0, 1.8] to be required to
    # match the masking behaviour in Iris regrid even though this is not expected to
    # influence the final result to the small overlap. Another solution is to decrease
    # the `tol` parameter of `get_overlapping()`, in this case below 3.55e-16
    # (e.g. 1e-16).
    # overlap_y = True

    src_y_chunks, tgt_y_slices = calculate_blocks(
        src_y_chunks,
        tgt_y_chunks,
        tgt_y_slices,
        min_src_chunk_size,
        max_src_chunk_size,
        min_tgt_chunk_size,
        max_tgt_chunk_size,
    )

    valid_src_slice = [slice(None)] * src_cube.ndim
    valid_src_slice[src_y_dim] = valid_src_y_slice
    valid_src_slice = tuple(valid_src_slice)

    # Re-chunk the data and coordinate along the y-dimension.
    block_src_data = src_cube.core_data()[valid_src_slice].rechunk(
        (
            src_y_chunks,
            -1,
        )
    )
    # 2D arrays are created here to enable consistent map_overlap behaviour.
    block_src_y_pnts = (
        da.from_array(src_y_coord.points[valid_src_y_slice])
        .rechunk((src_y_chunks,))
        .reshape(-1, 1)
    )
    block_src_low_y_bnds = (
        da.from_array(src_y_coord.bounds[valid_src_y_slice, 0])
        .rechunk((src_y_chunks,))
        .reshape(-1, 1)
    )
    block_src_upp_y_bnds = (
        da.from_array(src_y_coord.bounds[valid_src_y_slice, 1])
        .rechunk((src_y_chunks,))
        .reshape(-1, 1)
    )

    chunks_spec = [None] * 2
    chunks_spec[x_dim] = tgt_x_coord.shape[0]
    chunks_spec[y_dim] = tuple(slice_len(s) for s in tgt_y_slices)
    chunks_spec = tuple(chunks_spec)

    output = da.map_overlap(
        regrid_chunk,
        block_src_data,
        block_src_y_pnts,
        block_src_low_y_bnds,
        block_src_upp_y_bnds,
        # Store metadata for the y-coordinate for which points and bounds will be
        # filled in during the course of blocked regridding.
        src_y_coord_metadata=src_y_coord.metadata,
        src_x_coord=src_x_coord,
        src_cube_metadata=src_cube.metadata,
        tgt_y_coord=tgt_y_coord[valid_tgt_y_slice],
        tgt_x_coord=tgt_x_coord,
        tgt_y_slices=tgt_y_slices,
        tgt_cube_metadata=tgt_cube.metadata,
        y_dim=y_dim,
        x_dim=x_dim,
        scheme=scheme,
        depth={
            # The y-coordinate may need to be overlapped.
            y_dim: 1 if overlap_y else 0,
            # The x-coordinate will not be overlapped as it is never chunked.
            x_dim: 0,
        },
        boundary="none",
        trim=False,
        dtype=np.float64,
        chunks=chunks_spec,
        meta=np.array([], dtype=np.float64),
    )
    if not isinstance(output._meta, np.ma.MaskedArray):
        # XXX: Ideally this should not be needed, but the mask appears to vanish in
        # some cases.
        output = da.ma.masked_array(output, mask=False)

    if slice_len(valid_tgt_y_slice) < tgt_y_coord.shape[0]:
        # Embed the output data calculated above into the final target cube by padding
        # with masked data as necessary.
        seq = []
        if valid_tgt_y_slice.start > 0:
            # Pad at the start.
            start_pad_shape = [tgt_x_coord.shape[0]] * 2
            start_pad_shape[y_dim] = valid_tgt_y_slice.start
            seq.append(
                da.ma.masked_array(
                    da.zeros(start_pad_shape),
                    mask=True,
                )
            )
        seq.append(output)
        if valid_tgt_y_slice.stop < tgt_y_coord.shape[0]:
            # Pad at the end.
            end_pad_shape = [tgt_x_coord.shape[0]] * 2
            end_pad_shape[y_dim] = tgt_y_coord.shape[0] - valid_tgt_y_slice.stop
            seq.append(
                da.ma.masked_array(
                    da.zeros(end_pad_shape),
                    mask=True,
                )
            )
        output = da.concatenate(seq, axis=y_dim)

    return tgt_cube.copy(data=output)
