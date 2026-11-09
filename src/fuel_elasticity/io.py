"""Чтение шести листов кейса .xlsb (данные не попадают в git)."""
from pathlib import Path

import numpy as np
from pyxlsb import open_workbook

from .model import FuelDataset

REQUIRED_SHEETS = (
    "Данные", "Информация об АЗС", "Информация о клиентах",
    "Покупки сопутствующих товаров", "Маржа", "Плотность",
)


def _number(value: object, *, missing_to_zero: bool = False) -> float:
    if value is None:
        return 0.0 if missing_to_zero else float("nan")
    return float(value)


def _yes(value: object, expected: str = "1") -> bool:
    return str(value).strip().upper() == expected


def load_xlsb(path: str | Path) -> FuelDataset:
    """Импорт исходного файла чемпионата без изменения исходных сведений.

    Лист «Данные»: три строки заголовка, затем три блока по T колонок
    (объём / цена своей АЗС / цена конкурентов). Месяцы берутся из заголовка.
    Пропуски объёма (если они возникнут) не заменяются на нули молча.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Не найден XLSB: {path}")

    others: dict[str, list[list]] = {}
    ids, stations, fuels, volume, own, competitor = [], [], [], [], [], []
    with open_workbook(str(path)) as wb:
        missing = set(REQUIRED_SHEETS) - set(wb.sheets)
        if missing:
            raise ValueError(f"Отсутствуют листы: {sorted(missing)}")
        with wb.get_sheet("Данные") as sheet:
            for rowno, cells in enumerate(sheet.rows()):
                values = [cell.v for cell in cells]
                if rowno == 2:
                    if values[:3] != ["Клиент", "Основная АЗС заправки", "Топливо"]:
                        raise ValueError("Неожиданный формат листа «Данные»")
                    months = [str(m) for m in values[3:12]]
                    t = len(months)
                    if (values[3 + t:3 + 2*t] != values[3:3 + t]
                            or values[3 + 2*t:3 + 3*t] != values[3:3 + t]):
                        raise ValueError("Несовпадение месяцев между блоками данных")
                if rowno < 3:
                    continue
                if len(values) < 3 + 3*t or values[0] is None:
                    continue
                ids.append(str(values[0]))
                stations.append(str(values[1]))
                fuels.append(str(values[2]).upper())
                volume.append([_number(x) for x in values[3:3+t]])
                own.append([_number(x) for x in values[3+t:3+2*t]])
                competitor.append([_number(x) for x in values[3+2*t:3+3*t]])

        for name in REQUIRED_SHEETS[1:]:
            with wb.get_sheet(name) as sheet:
                rows = []
                for cells in sheet.rows():
                    row = [c.v for c in cells]
                    if any(value is not None for value in row):
                        rows.append(row)
                others[name] = rows

    # «Подразделение» — операционное региональное поле в справочнике АЗС, индекс 5.
    station_lookup = {str(row[0]): row for row in others["Информация об АЗС"][1:]}
    client_lookup = {str(row[0]): row for row in others["Информация о клиентах"][1:]}
    goods_lookup = {str(row[0]): row for row in others["Покупки сопутствующих товаров"][1:]}
    margin_lookup = {(str(row[1]), str(row[0]).upper()): row
                     for row in others["Маржа"][1:]}
    density_lookup = {str(row[0]).upper(): float(row[1])
                      for row in others["Плотность"][1:]}

    missing_stations = set(stations) - station_lookup.keys()
    if missing_stations:
        raise ValueError(f"АЗС не найдены в справочнике: {len(missing_stations)}")

    regions = [str(station_lookup[s][5]) for s in stations]
    missing_margin = [(r, f) for r, f in zip(regions, fuels)
                      if (r, f) not in margin_lookup or f not in density_lookup]
    if missing_margin:
        raise ValueError(f"Нет финансовых параметров у {len(missing_margin)} клиентов")

    # Плотность задана в т/м³. Масса 1 л = density/1000 тонны.
    net_unit = np.array([
        (float(margin_lookup[(r, f)][2]) - float(margin_lookup[(r, f)][3]))
        * density_lookup[f] / 1000
        for r, f in zip(regions, fuels)
    ], dtype=float)
    dataset = FuelDataset(
        ids=ids, stations=stations, fuels=fuels, regions=regions, months=months,
        volume=np.asarray(volume, dtype=float), own_price=np.asarray(own, dtype=float),
        competitor_price=np.asarray(competitor, dtype=float),
        app=np.array([_yes(client_lookup[k][5]) if k in client_lookup else False
                      for k in ids], dtype=bool),
        push=np.array([_yes(client_lookup[k][7]) if k in client_lookup else False
                       for k in ids], dtype=bool),
        sms=np.array([_yes(client_lookup[k][4], "Y") if k in client_lookup else False
                      for k in ids], dtype=bool),
        goods_revenue=np.array([
            _number(goods_lookup[k][4], missing_to_zero=True) if k in goods_lookup else 0.
            for k in ids
        ]),
        contribution_per_liter=net_unit,
    )
    dataset.validate()
    return dataset
