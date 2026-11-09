"""Структуры данных и проверки корректности матриц наблюдений."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FuelDataset:
    ids: list[str]
    stations: list[str]
    fuels: list[str]
    regions: list[str]
    months: list[str]
    volume: np.ndarray
    own_price: np.ndarray
    competitor_price: np.ndarray
    app: np.ndarray
    push: np.ndarray
    sms: np.ndarray
    goods_revenue: np.ndarray
    contribution_per_liter: np.ndarray

    def validate(self) -> None:
        n, t = self.volume.shape
        if n == 0 or t < 4:
            raise ValueError("Необходимо не менее одного клиента и четырёх месяцев")
        if not (len(self.ids) == len(self.stations) == len(self.fuels)
                == len(self.regions) == n):
            raise ValueError("Неправильная длина справочников клиентов")
        if len(self.months) != t or len(set(self.months)) != t:
            raise ValueError("Некорректные месяцы")
        if len(set(self.ids)) != n:
            raise ValueError("Есть повторяющиеся ID клиентов — агрегацию следует проверить")
        if self.own_price.shape != (n, t) or self.competitor_price.shape != (n, t):
            raise ValueError("Цена и объём должны иметь одинаковые размеры")
        for key in ("volume", "own_price", "competitor_price"):
            values = getattr(self, key)
            if not np.isfinite(values).all():
                raise ValueError(f"В матрице {key} имеются пропуски / NaN / Inf")
            if (values < 0).any():
                raise ValueError(f"В матрице {key} имеются отрицательные значения")
        # В оригинальном кейсе 473 парных нулевых значения обеих цен.
        # Для воспроизведения исторического расчёта такие пары допускаются
        # (разрыв становится 0), но включаются в диагностические предупреждения.
        if np.any((self.own_price == 0) != (self.competitor_price == 0)):
            raise ValueError("Нулевая цена есть только у одного из конкурентов")
        for key in ("app", "push", "sms", "goods_revenue", "contribution_per_liter"):
            if getattr(self, key).shape != (n,):
                raise ValueError(f"Поле {key} должно быть длины {n}")
        if not np.isfinite(self.contribution_per_liter).all():
            raise ValueError("Не найдены параметры маржи/плотности для части клиентов")

    @property
    def gap(self) -> np.ndarray:
        """Положительное значение: наша АЗС дороже средней цены конкурентов."""
        return self.own_price - self.competitor_price
