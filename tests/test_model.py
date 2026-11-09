from dataclasses import replace

import numpy as np
import pytest

from fuel_elasticity.demo import make_demo_dataset


def test_demo_is_deterministic_and_well_formed():
    a = make_demo_dataset(120, seed=7)
    b = make_demo_dataset(120, seed=7)
    assert a.volume.shape == (120, 9)
    np.testing.assert_allclose(a.gap, b.gap)
    assert a.months[0] == "2023-09"
    assert a.months[-1] == "2024-05"


def test_duplicate_ids_rejected():
    a = make_demo_dataset(120, seed=7)
    with pytest.raises(ValueError, match="повторяющиеся ID"):
        replace(a, ids=["SAME"] * 120).validate()


def test_one_sided_price_zero_rejected():
    a = make_demo_dataset(120, seed=7)
    own = a.own_price.copy()
    own[0, 0] = 0
    with pytest.raises(ValueError, match="Нулевая цена"):
        replace(a, own_price=own).validate()


def test_matching_zero_prices_allowed_for_reproducibility():
    a = make_demo_dataset(120, seed=7)
    own = a.own_price.copy()
    comp = a.competitor_price.copy()
    own[0, 0] = comp[0, 0] = 0
    replace(a, own_price=own, competitor_price=comp).validate()
