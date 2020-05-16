# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from wildfires.analysis.analysis import _get_constraints_title, _match_constraints
from wildfires.data import VariableNotFoundError


def test_match_constraints():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 10, 20]})

    constraints = ({"A": slice(None)},)
    selection = _match_constraints(constraints, X)
    assert np.all(selection == np.array([1, 1, 1, 1, 1], dtype=np.bool_))

    constraints = ({"A": slice(1, 3), "B": slice(None, 4)},)
    selection = _match_constraints(constraints, X)
    assert np.all(selection == np.array([1, 0, 0, 0, 0], dtype=np.bool_))

    constraints = ({"A": slice(2, None), "B": slice(None, 20)},)
    selection = _match_constraints(constraints, X)
    assert np.all(selection == np.array([0, 1, 1, 1, 0], dtype=np.bool_))

    constraints = ({"A": slice(2, None)},)
    selection = _match_constraints(constraints, X)
    assert np.all(selection == np.array([0, 1, 1, 1, 1], dtype=np.bool_))

    constraints = ({"A": slice(2, 4)}, {"B": slice(5, None)})
    selection = _match_constraints(constraints, X)
    assert np.all(selection == np.array([0, 1, 1, 1, 1], dtype=np.bool_))

    with pytest.raises(VariableNotFoundError):
        constraints = ({"C": slice(None)},)
        selection = _match_constraints(constraints, X)


def test_constraints_title():
    constraints = ({"A": slice(1, 2), "B": slice(None, 2)},)
    title = _get_constraints_title(constraints)
    assert title == "1 < A < 2 & B < 2"

    constraints = ({"A": slice(1, 2), "B": slice(None, 2)}, {"A": slice(3)})
    title = _get_constraints_title(constraints)
    assert title == "(1 < A < 2 & B < 2) | (A < 3)"
