#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import pytest

import wildfires.data.datasets as wildfire_datasets
from test_datasets import data_availability
from wildfires.data.cube_aggregation import Selection


@pytest.fixture(scope="module")
def sel():
    sel = Selection()
    sel.add("HYDE", ("popd", "pop density"))
    sel.add("GFEDv4", ("monthly burned area", "burned area"))
    sel.add("Dataset", "raw_name")
    return sel


@pytest.fixture(scope="module")
def long_sel():
    long_sel = Selection()
    long_sel.add("HYDE", ("popd", "pop density"))
    long_sel.add("HYDE", ("grazing land area", "grazing area"))
    long_sel.add("GFEDv4", ("monthly burned area", "burned area"))
    long_sel.add("GFEDv4", ("monthly average co2 emissions", "co2 emissions"))
    long_sel.add("Dataset", "raw_name")
    return long_sel


def test_representations(sel):
    # Confirm expected output.
    all_all = sel.get(dataset="all", variable_format="all")
    assert all_all == {
        ("HYDE", "HYDE", None): (("popd", "pop density"),),
        ("GFEDv4", "GFEDv4", None): (("monthly burned area", "burned area"),),
        ("Dataset", "Dataset", None): (("raw_name", "raw_name"),),
    }

    # Confirm correct caching.
    assert all_all == sel._Selection__formatted["all"]

    # Confirm expected output.
    all_pretty = sel.get(dataset="all", variable_format="pretty")
    assert all_pretty == {
        "HYDE": ("pop density",),
        "GFEDv4": ("burned area",),
        "Dataset": ("raw_name",),
    }

    # Confirm correct caching.
    assert all_pretty == sel._Selection__formatted["pretty"]

    # Confirm expected output.
    all_raw = sel.get(dataset="all", variable_format="raw")
    assert all_raw == {
        "HYDE": ("popd",),
        "GFEDv4": ("monthly burned area",),
        "Dataset": ("raw_name",),
    }

    # Confirm correct caching.
    assert all_raw == sel._Selection__formatted["raw"]


def test_adding(sel):
    # Test guard against duplicated names.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "raw variable name 'popd' is already present in {Dataset(raw='HYDE', "
            "pretty='HYDE', instance=None): [Variable(raw='popd', "
            "pretty='pop density')]}"
        ),
    ):
        sel.add("HYDE", "popd")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "pretty variable name 'pop density' is already present in "
            "{Dataset(raw='HYDE', pretty='HYDE', instance=None): "
            "[Variable(raw='popd', pretty='pop density')]}"
        ),
    ):
        sel.add("HYDE", ("popd2", "pop density"))


def test_duplication():

    with pytest.raises(ValueError):
        Selection().add("A", ("a", "b")).add("A", ("a", "b"))

    with pytest.raises(ValueError):
        Selection().add("A", ("a", "b")).add("A", ("a"))

    with pytest.raises(ValueError):
        Selection().add("A", ("a", "b")).add("A", ("a", "0"))

    with pytest.raises(ValueError):
        Selection().add("A", ("a", "b")).add("A", ("0", "b"))

    with pytest.raises(ValueError):
        Selection().add(("A", "abc"), "b").add(("B", "abc"), "c")

    assert isinstance(Selection().add("A", ("a", "b")).add("A", ("c", "d")), Selection)

    with pytest.raises(ValueError):
        Selection().add("A", "a").add("B", "a").select_variables("a")
        Selection().add("A", "a").add("B", "a").select_variables("0")

    assert isinstance(
        Selection().add("A", "a").add("B", "a").dict_select_variables({"A": ("a",)}),
        Selection,
    )

    assert isinstance(
        Selection().add("A", "a").add("B", "a").dict_select_variables({"B": ("a",)}),
        Selection,
    )

    with pytest.raises(KeyError):
        Selection().add("A", "a").add("B", "a").dict_select_variables({"C": ("a",)})

    with pytest.raises(KeyError):
        Selection().add("A", "a").add("B", "a").dict_select_variables({"A": ("b",)})


def test_name_retrieval(sel):
    """Test that all names are retrieved correctly."""
    assert set(sel.raw_variable_names) == set(
        ("popd", "monthly burned area", "raw_name")
    )
    assert set(sel.pretty_variable_names) == set(
        ("pop density", "burned area", "raw_name")
    )


def test_equality(sel, long_sel):
    sel2 = Selection()
    sel2.add("GFEDv4", ("monthly burned area", "burned area"))
    sel2.add("HYDE", ("popd", "pop density"))
    sel2.add("Dataset", "raw_name")

    assert sel2 == sel

    # If we leave out the pretty names, the two Selections should no longer be equal.
    sel3 = Selection()
    sel3.add("GFEDv4", "monthly burned area")
    sel3.add("HYDE", "popd")
    sel3.add("Dataset", "raw_name")

    assert sel3 != sel

    # See if different variable assignment orders affect equality.

    long_sel2 = Selection()
    long_sel2.add("HYDE", ("grazing land area", "grazing area"))
    long_sel2.add("HYDE", ("popd", "pop density"))
    long_sel2.add("GFEDv4", ("monthly average co2 emissions", "co2 emissions"))
    long_sel2.add("GFEDv4", ("monthly burned area", "burned area"))
    long_sel2.add("Dataset", "raw_name")

    assert long_sel == long_sel2


def test_hash(sel):
    sel2 = Selection()
    sel2.add("GFEDv4", ("monthly burned area", "burned area"))
    sel2.add("HYDE", ("popd", "pop density"))
    sel2.add("Dataset", "raw_name")

    assert len(set((sel2, sel))) == 1

    # Leaving out the pretty variable names should yield a different result.

    sel3 = Selection()
    sel3.add("GFEDv4", "monthly burned area")
    sel3.add("HYDE", "popd")
    sel3.add("Dataset", "raw_name")

    assert len(set((sel3, sel))) != 1


def test_selection(sel, long_sel):
    # Selection using a dataset: name dict.
    assert sel.select_variables(("burned area", "popd", "raw_name")) == sel

    # # Testing regex support as well.
    assert sel.select_variables(("burned", "pop", "raw_"), exact=False) == sel
    assert sel.select_variables(("b.*area", "pop", "r.*name"), exact=False) == sel

    with pytest.raises(KeyError, match=re.escape("Variable 'bu*area' not found.")):
        sel.select_variables(("bu*area",), exact=False)

    with pytest.raises(KeyError, match=re.escape("Variable 'b+area' not found.")):
        sel.select_variables(("b+area",), exact=False)

    assert sel.select_variables(("urned", "pop", "raw_"), exact=False) == sel
    with pytest.raises(KeyError, match=re.escape("Variable 'urned' not found.")):
        sel.select_variables(("urned", "pop", "raw_"), exact=True)

    assert set(
        sel.select_variables(("burned area", "raw_name")).pretty_variable_names
    ) == set(("burned area", "raw_name"))
    assert sel.dict_select_variables({"HYDE": ("popd",)}) == Selection().add(
        "HYDE", ("popd", "pop density")
    )
    assert sel.dict_select_variables({"HYDE": ("pop density",)}) == Selection().add(
        "HYDE", ("popd", "pop density")
    )

    assert long_sel.dict_select_variables(
        {"HYDE": ("pop density", "grazing area"), "Dataset": ("raw_name",)}
    ) == Selection().add("HYDE", ("popd", "pop density")).add(
        "HYDE", ("grazing land area", "grazing area")
    ).add(
        "Dataset", "raw_name"
    )


def test_removal(sel, long_sel):
    assert sel.remove_variables(
        ("burned area", "raw_name"), inplace=False
    ) == sel.select_variables("popd")

    assert set(
        sel.remove_variables("raw_name", inplace=False).raw_dataset_names
    ) == set(("HYDE", "GFEDv4"))

    assert sel.remove_datasets(
        ("Dataset", "HYDE"), inplace=False
    ) == sel.select_datasets("GFEDv4")


def test_creation(sel):
    comp_sel = Selection(
        {
            "HYDE": [("popd", "pop density")],
            "GFEDv4": [("monthly burned area", "burned area")],
            "Dataset": ["raw_name"],
        }
    )
    assert comp_sel == sel


def test_addition(sel):
    test_sel = Selection()
    orig_id = id(test_sel)
    test_sel += {"HYDE": [("popd", "pop density")]}
    test_sel += Selection().add("GFEDv4", ("monthly burned area", "burned area"))
    test_sel += Selection().add("Dataset", "raw_name")

    print(repr(test_sel))
    print(repr(sel))
    assert test_sel == sel
    assert id(test_sel) == orig_id

    with pytest.raises(
        ValueError,
        match=re.escape(
            "raw variable name 'popd' is already present in {Dataset(raw='HYDE', "
            "pretty='HYDE', instance=None): [Variable(raw='popd', "
            "pretty='pop density')]}"
        ),
    ):
        test_sel.add("HYDE", ("popd", "pop density"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "raw variable name 'popd' is already present in {Dataset(raw='HYDE', "
            "pretty='HYDE', instance=None): [Variable(raw='popd', "
            "pretty='pop density')]}"
        ),
    ):
        _ = test_sel + sel

    test_sel2 = Selection()
    orig_id2 = id(test_sel2)
    test_sel2 = test_sel2 + {"HYDE": [("popd", "pop density")]}
    test_sel2 = test_sel2 + Selection().add(
        "GFEDv4", ("monthly burned area", "burned area")
    )
    test_sel2 = test_sel2 + Selection().add("Dataset", "raw_name")

    assert test_sel2 == sel
    assert id(test_sel2) != orig_id2


@data_availability
def test_instances():
    hyde = wildfire_datasets.HYDE()
    sel1 = Selection().add({"raw": "HYDE", "instance": hyde}, "popd")
    sel2 = Selection().add({"raw": "HYDE", "instance": hyde}, "popd")

    assert sel1 == sel2
    assert sel1.instances == sel2.instances

    sel3 = Selection().add(
        {"raw": "HYDE", "instance": wildfire_datasets.HYDE()}, "popd"
    )

    assert sel1 != sel3

    agb = wildfire_datasets.AvitabileThurnerAGB()

    multi_sel = sel1.copy()

    assert multi_sel != sel1
    assert multi_sel.get("all", "raw") == sel1.get("all", "raw")
    assert multi_sel.get("all", "pretty") == sel1.get("all", "pretty")
    assert multi_sel is not sel1

    multi_sel.add({"raw": "AGB", "instance": agb}, "agb")

    names = multi_sel.raw_dataset_names
    instances = multi_sel.instances

    assert names.index("AGB") == instances.index(agb)
    # Need to get reference to hyde instance anew here, since the stored instance
    # won't match the hyde instance in `sel1`, as `sel1.copy()` was used to create
    # `multi_sel`.
    assert names.index("HYDE") == instances.index(multi_sel.get_index("HYDE").instance)


def test_pack_input():
    pack_input = Selection.pack_input

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expected items to be of type(s) '{}', but got types "
            "'[{}]'.".format(str, type(None))
        ),
    ):
        pack_input((None,))

    with pytest.raises(
        ValueError, match=re.escape("Expected at most 1 item(s), got 2.")
    ):
        pack_input(("test1", "test2"), elements=1)


def cache_is_empty(selection):
    return all(
        not selection._Selection__return_cached_repr("all", variable_format)
        for variable_format in ("all", "raw", "pretty")
    )


def test_pruning():
    assert Selection().add("A", "a").remove_variables("a") == Selection()
    assert (
        Selection().add("A", "a").add("A", "b").remove_variables(("a", "b"))
        == Selection()
    )
    assert Selection().add("A", "a").add("A", "b").remove_variables(
        "b"
    ) == Selection().add("A", "a")
    assert Selection().add("A", "a").remove_datasets("A") == Selection()

    assert cache_is_empty(
        Selection().add("A", "a").add("A", "b").remove_variables(("a", "b"))
    )
