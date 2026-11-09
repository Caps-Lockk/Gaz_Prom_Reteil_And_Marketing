"""Непричинная сегментация по условной реакции на относительную цену."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .model import FuelDataset


@dataclass(frozen=True)
class SegmentationRules:
    min_active_months: int = 6
    min_liters: float = 250.0
    min_residual_gap_std: float = 0.20
    sensitive_max_beta: float = -0.15
    sensitive_max_correlation: float = -0.35
    sensitive_min_negative_loo: int = 8
    stable_min_beta: float = -0.04
    stable_min_correlation: float = -0.15
    stable_min_positive_loo: int = 5


@dataclass(frozen=True)
class CustomerScores:
    beta: np.ndarray
    correlation: np.ndarray
    residual_gap_std: np.ndarray
    loo_negative: np.ndarray
    loo_positive: np.ndarray
    active_months: np.ndarray
    total_liters: np.ndarray
    qualified: np.ndarray
    sensitive: np.ndarray
    stable: np.ndarray

    @property
    def segment(self) -> np.ndarray:
        return np.where(self.sensitive, "Ценочувствительный",
                        np.where(self.stable, "Ценоустойчивый", "Неопределённый"))


def residualize_by_region_fuel(
    x: np.ndarray, y: np.ndarray, regions: list[str], fuels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Двойная центровка: индивидуальные FE и (регион × топливо × месяц) FE.

    Применяется независимо для каждого временного окна (train / holdout).
    """
    if x.shape != y.shape or x.shape[0] != len(regions) or len(regions) != len(fuels):
        raise ValueError("Неверные формы данных при центровке")
    xr = np.empty_like(x, dtype=float)
    yr = np.empty_like(y, dtype=float)
    groups = defaultdict(list)
    for i, key in enumerate(zip(regions, fuels)):
        groups[key].append(i)
    for indices in groups.values():
        idx = np.asarray(indices, dtype=int)
        X, Y = x[idx], y[idx]
        xr[idx] = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()
        yr[idx] = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + Y.mean()
    return xr, yr


def score_clients(
    ds: FuelDataset,
    months: slice | None = None,
    rules: SegmentationRules | None = None,
) -> CustomerScores:
    """β_i: наклон log(1+литры) по разрыву цен после двойной центровки."""
    rules = rules or SegmentationRules()
    selected = months if months is not None else slice(None)
    v, gap = ds.volume[:, selected], ds.gap[:, selected]
    n, t = v.shape
    if t < 4:
        raise ValueError("Меньше четырёх месяцев для сегментации")
    X, Y = residualize_by_region_fuel(gap, np.log1p(v), ds.regions, ds.fuels)
    xx = np.sum(X**2, axis=1)
    xy = np.sum(X * Y, axis=1)
    yy = np.sum(Y**2, axis=1)
    beta = np.divide(xy, xx, out=np.full(n, np.nan), where=xx > 1e-10)
    corr = np.divide(xy, np.sqrt(xx * yy), out=np.full(n, np.nan),
                     where=(xx * yy) > 1e-10)
    negative = np.zeros(n, dtype=int)
    positive = np.zeros(n, dtype=int)
    for j in range(t):
        x2 = np.delete(X, j, axis=1)
        y2 = np.delete(Y, j, axis=1)
        x2 -= x2.mean(axis=1, keepdims=True)
        y2 -= y2.mean(axis=1, keepdims=True)
        denominator = np.sum(x2**2, axis=1)
        slope = np.divide(np.sum(x2*y2, axis=1), denominator,
                          out=np.zeros(n), where=denominator > 1e-10)
        negative += slope < 0
        positive += slope > 0
    active = np.sum(v > 0, axis=1)
    total = v.sum(axis=1)
    std = np.sqrt(xx / t)
    qual = (active >= rules.min_active_months) & (std >= rules.min_residual_gap_std) & (
        total >= rules.min_liters)
    sensitive = (qual & (beta <= rules.sensitive_max_beta)
                 & (corr <= rules.sensitive_max_correlation)
                 & (negative >= rules.sensitive_min_negative_loo))
    stable = (qual & (beta >= rules.stable_min_beta)
              & (corr >= rules.stable_min_correlation)
              & (positive >= rules.stable_min_positive_loo))
    return CustomerScores(beta, corr, std, negative, positive, active, total,
                          qual, sensitive, stable)
