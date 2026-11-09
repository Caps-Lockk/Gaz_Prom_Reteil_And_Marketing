from dataclasses import replace

import numpy as np
import pytest

from fuel_elasticity.demo import make_demo_dataset
from fuel_elasticity.economics import OfferPolicy, calculate_offer, eligible_for_offer
from fuel_elasticity.segmentation import CustomerScores, score_clients


def test_subsidize_entire_baseline_volume_not_just_incremental():
    original = make_demo_dataset(120, seed=2)
    volume = np.full((120, 9), 50.0)
    own_price = np.full((120, 9), 54.0)
    competitor_price = np.full((120, 9), 52.0)
    ds = replace(original, volume=volume, own_price=own_price,
                 competitor_price=competitor_price,
                 app=np.ones(120, dtype=bool), push=np.ones(120, dtype=bool),
                 contribution_per_liter=np.full(120, 10.0))
    n = 120
    sc = CustomerScores(np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, int),
                        np.zeros(n, int), np.full(n, 9), np.full(n, 450),
                        np.ones(n, bool), np.ones(n, bool), np.zeros(n, bool))
    result = calculate_offer(ds, sc, OfferPolicy(.5, 70, .1, .5))
    # 50 л → 55 л, вклад 5×10=50, расходы 55×0,5=27,5 на клиента
    assert result["eligible_clients"] == n
    assert result["incremental_contribution_rub"] == pytest.approx(50*n)
    assert result["bonus_cost_rub"] == pytest.approx(27.5*n)
    assert result["net_incremental_rub"] == pytest.approx(22.5*n)


def test_policy_requires_consent_and_price_disadvantage():
    ds = make_demo_dataset(1200, seed=42)
    scores = score_clients(ds)
    policy = OfferPolicy()
    mask = eligible_for_offer(ds, scores, policy)
    assert np.all(ds.app[mask] & ds.push[mask])
    assert np.all(ds.gap[mask, -1] > .5)
    denied = replace(ds, push=np.zeros(len(ds.ids), dtype=bool))
    assert not eligible_for_offer(denied, scores, policy).any()


def test_invalid_policy():
    ds = make_demo_dataset(120, seed=1)
    with pytest.raises(ValueError):
        calculate_offer(ds, score_clients(ds), OfferPolicy(rebate_per_liter=-1))
