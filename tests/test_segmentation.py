import numpy as np

from fuel_elasticity.demo import make_demo_dataset
from fuel_elasticity.segmentation import residualize_by_region_fuel, score_clients
from fuel_elasticity.validation import temporal_holdout


def test_residualization_removes_client_mean_and_group_month_mean():
    ds = make_demo_dataset(180, seed=3)
    x, y = residualize_by_region_fuel(ds.gap, np.log1p(ds.volume), ds.regions, ds.fuels)
    assert np.max(abs(x.mean(axis=1))) < 1e-9
    assert np.max(abs(y.mean(axis=1))) < 1e-9
    for group in set(zip(ds.regions, ds.fuels)):
        idx = np.array([i for i, k in enumerate(zip(ds.regions, ds.fuels)) if k == group])
        assert np.max(abs(x[idx].mean(axis=0))) < 1e-9


def test_segment_counts_exhaustive_and_exclusive():
    ds = make_demo_dataset(1200, seed=42)
    scores = score_clients(ds)
    assert np.sum(scores.sensitive & scores.stable) == 0
    assert sum(np.unique(scores.segment, return_counts=True)[1]) == 1200
    assert scores.sensitive.sum() > 0
    assert scores.stable.sum() > 0
    assert scores.beta.shape == (1200,)


def test_holdout_does_not_use_future_for_segmenting():
    ds = make_demo_dataset(1200, seed=42)
    a = temporal_holdout(ds)
    assert sum(x["clients"] for x in a["groups"].values()) == 1200
    assert len(a["train_period"]) == 7 and len(a["test_period"]) == 2
    assert a["groups"]["sensitive"]["holdout_beta"] < a["groups"]["stable"]["holdout_beta"]
