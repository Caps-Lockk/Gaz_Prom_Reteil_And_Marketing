"""Честный temporal holdout: 7 месяцев для сегментации, 2 для проверки."""

import numpy as np

from .model import FuelDataset
from .segmentation import SegmentationRules, residualize_by_region_fuel, score_clients


def temporal_holdout(ds: FuelDataset) -> dict:
    """Возвращает групповые наклоны на отложенных месяцах.

    Для семи тренировочных месяцев используются слегка более мягкие пороги
    активности и leave-one-out (5 месяцев, 200 л, >=6/7 отрицательных знаков).
    """
    if ds.volume.shape[1] != 9:
        raise ValueError("Проверка 7+2 ожидает ровно 9 месяцев исходного кейса")
    train_rules = SegmentationRules(
        min_active_months=5, min_liters=200.0,
        sensitive_min_negative_loo=6,
        stable_min_positive_loo=5,
    )
    train = score_clients(ds, slice(0, 7), train_rules)
    # Сохранение точного исторического правила: устойчивая группа <=2
    # отрицательных LOO, т.е. >=5 положительных из 7 (при ненулевых наклонах).
    X, Y = residualize_by_region_fuel(
        ds.gap[:, 7:9], np.log1p(ds.volume[:, 7:9]), ds.regions, ds.fuels)
    masks = {"sensitive": train.sensitive,
             "stable": train.stable,
             "uncertain": ~(train.sensitive | train.stable)}
    summary = {}
    for name, mask in masks.items():
        denom = float(np.sum(X[mask] ** 2))
        beta = float(np.sum(X[mask]*Y[mask])/denom) if denom > 0 else None
        summary[name] = {"clients": int(mask.sum()), "holdout_beta": beta}
    return {"train_period": ds.months[:7], "test_period": ds.months[7:9],
            "groups": summary}
