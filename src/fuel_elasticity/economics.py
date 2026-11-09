"""Экономика персонального возврата на литр: модель, не результат A/B-теста."""
from dataclasses import asdict, dataclass

import numpy as np

from .model import FuelDataset
from .segmentation import CustomerScores


@dataclass(frozen=True)
class OfferPolicy:
    rebate_per_liter: float = 0.50
    cap_liters: float = 70.0
    assumed_uplift: float = 0.05
    min_price_gap: float = 0.50

    def validate(self) -> None:
        if self.rebate_per_liter < 0 or self.cap_liters <= 0:
            raise ValueError("Скидка не может быть отрицательной, лимит должен быть > 0")
        if self.assumed_uplift < 0 or self.min_price_gap < 0:
            raise ValueError("Прирост и порог не могут быть отрицательными")


def eligible_for_offer(
    ds: FuelDataset, scores: CustomerScores, policy: OfferPolicy
) -> np.ndarray:
    """Оффер: чувствительный, активный за 3 мес., app+push, цена > конкурентов."""
    policy.validate()
    return (scores.sensitive & (ds.volume[:, -3:].sum(axis=1) > 0)
            & ds.app & ds.push & (ds.gap[:, -1] > policy.min_price_gap))


def calculate_offer(
    ds: FuelDataset, scores: CustomerScores, policy: OfferPolicy
) -> dict:
    """Чистый эффект = дополнительный вклад − бонусы на ВЕСЬ объём в лимите.

    Важно: бонусы есть также для покупок, которые состоялись бы и без оффера.
    Дополнительные литры и маржинальность по ним здесь сценарные предположения.
    """
    mask = eligible_for_offer(ds, scores, policy)
    baseline = ds.volume[mask, -1]
    contrib = ds.contribution_per_liter[mask]
    scenario_liters = baseline * (1.0 + policy.assumed_uplift)
    incremental_liters = scenario_liters - baseline
    reward_liters = np.minimum(scenario_liters, policy.cap_liters)
    bonus_cost = float(np.sum(reward_liters * policy.rebate_per_liter))
    extra_contribution = float(np.sum(incremental_liters * contrib))
    return {
        "offer_policy": asdict(policy),
        "eligible_clients": int(mask.sum()),
        "baseline_liters": float(baseline.sum()),
        "scenario_liters": float(scenario_liters.sum()),
        "incremental_liters": float(incremental_liters.sum()),
        "bonus_cost_rub": bonus_cost,
        "incremental_contribution_rub": extra_contribution,
        "net_incremental_rub": extra_contribution - bonus_cost,
        "breakeven_uplift": _breakeven(baseline, contrib, policy),
    }


def _breakeven(baseline: np.ndarray, contribution: np.ndarray,
               policy: OfferPolicy) -> float | None:
    if baseline.size == 0 or np.sum(baseline * contribution) <= 0:
        return None
    def profit(u: float) -> float:
        return (float(np.sum(baseline * contribution * u))
                - float(np.sum(np.minimum(baseline*(1+u), policy.cap_liters)
                               * policy.rebate_per_liter)))
    if profit(0.0) >= 0:
        return 0.0
    hi = 0.05
    while profit(hi) < 0 and hi < 100.0:
        hi *= 2
    if profit(hi) < 0:
        return None
    lo = 0.0
    for _ in range(70):
        mid = (hi + lo)/2
        if profit(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi
