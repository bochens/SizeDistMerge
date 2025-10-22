# tests/test_edges_from_centers.py

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from dev.AerosolSizedist import AerosolSizedist


def test_edges_from_centers_basic():
    c = np.array([10.0, 20.0, 40.0], dtype=float)
    lower, upper = AerosolSizedist._edges_from_centers(c)

    # expected edges via geometric means (nm)
    edges_expected = np.array(
        [7.0710678118654755, 14.142135623730951, 28.284271247461902, 56.568542494923804]
    )
    assert lower.shape == c.shape
    assert upper.shape == c.shape
    assert np.allclose(lower, edges_expected[:-1])
    assert np.allclose(upper, edges_expected[1:])
    assert np.allclose(np.sqrt(lower * upper), c)


def test_edges_from_centers_error_on_short_input():
    c = np.array([100.0], dtype=float)
    try:
        AerosolSizedist._edges_from_centers(c)
        assert False, "expected ValueError"
    except ValueError:
        assert True