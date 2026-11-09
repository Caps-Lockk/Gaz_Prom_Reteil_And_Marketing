"""Синтетический набор, не использующий приватные данные кейса."""

import numpy as np

from .model import FuelDataset


def make_demo_dataset(n: int = 1200, seed: int = 42) -> FuelDataset:
    if n < 60:
        raise ValueError("Для демо требуется не менее 60 клиентов")
    rng = np.random.default_rng(seed)
    months = ["2023-09", "2023-10", "2023-11", "2023-12", "2024-01",
              "2024-02", "2024-03", "2024-04", "2024-05"]
    station_idx = rng.integers(0, 36, n)
    region_idx = station_idx % 4
    fuel_idx = rng.integers(0, 2, n)
    regions = [f"DEMO_REGION_{k}" for k in region_idx]
    fuels = ["АИ-92" if k == 0 else "АИ-95" for k in fuel_idx]
    # Станционный price-shock — вариация цен внутри одного регионального месяца.
    shock = rng.normal(0, 0.95, (36, 9))
    trend = np.linspace(-0.45, 1.45, 9)
    gap = trend[np.newaxis, :] + shock[station_idx] + rng.normal(0, .04, (n, 9))
    competitors = 52 + np.linspace(0, 1.2, 9)[None, :] + fuel_idx[:, None] * 4
    competitor_price = competitors + np.zeros((n, 9))
    own_price = competitor_price + gap
    # Синтетический ground truth: три механизма реакции (не эквивалентны
    # сегментам, которые восстановит исследовательская модель).
    true_group = np.arange(n) % 3
    slope = np.where(true_group == 0, -.52, np.where(true_group == 1, -.01, -.14))
    user_base = rng.uniform(75, 200, n)
    seasonality = np.array([0.12, .09, .05, .02, -.10, -.12, -.08, -.03, .02])
    log_v = (np.log(user_base)[:, None]
             + slope[:, None] * (gap - gap.mean(axis=1, keepdims=True))
             + seasonality[None, :] + rng.normal(0, .09, (n, 9)))
    volume = np.maximum(np.expm1(log_v), 0.0)
    app = rng.random(n) < .8
    push = rng.random(n) < .85
    sms = rng.random(n) < .75
    goods = np.where(rng.random(n) < .7, rng.uniform(80, 1600, n), 0)
    ds = FuelDataset(
        ids=[f"SYNTH_{i+1:05d}" for i in range(n)],
        stations=[f"DEMO_STATION_{k:03d}" for k in station_idx],
        fuels=fuels, regions=regions, months=months, volume=volume,
        own_price=own_price, competitor_price=competitor_price, app=app,
        push=push, sms=sms, goods_revenue=goods,
        contribution_per_liter=np.where(fuel_idx == 0, 9.5, 10.5).astype(float),
    )
    ds.validate()
    return ds
