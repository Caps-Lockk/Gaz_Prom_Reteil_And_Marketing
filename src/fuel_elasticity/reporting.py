"""Обезличенные агрегаты для публикации и закрытая клиентская выгрузка."""
from collections import Counter
from pathlib import Path
import csv
import json

import numpy as np

from .economics import OfferPolicy, calculate_offer, eligible_for_offer
from .model import FuelDataset
from .segmentation import score_clients
from .validation import temporal_holdout


def build_summary(ds: FuelDataset, policy: OfferPolicy) -> tuple[dict, object]:
    scores = score_clients(ds)
    offer = calculate_offer(ds, scores, policy)
    counts = Counter(scores.segment.tolist())
    gap = ds.gap
    summary = {
        "dataset": {
            "clients": len(ds.ids),
            "unique_main_stations": len(set(ds.stations)),
            "period": [ds.months[0], ds.months[-1]],
            "total_liters": float(ds.volume.sum()),
            "zero_client_month_share": float(np.mean(ds.volume == 0)),
            "customer_months": int(ds.volume.size),
            "paired_zero_price_months": int(np.sum((ds.own_price == 0)
                                                     & (ds.competitor_price == 0))),
        },
        "monthly": [
            {"month": month, "liters": float(ds.volume[:, j].sum()),
             "mean_price_gap_rub_per_liter": float(gap[:, j].mean())}
            for j, month in enumerate(ds.months)
        ],
        "segments": {
            segment: {"clients": int(counts[segment]),
                      "share": counts[segment]/len(ds.ids),
                      "liters": float(ds.volume[scores.segment == segment].sum())}
            for segment in ("Ценочувствительный", "Ценоустойчивый", "Неопределённый")
        },
        "validation": temporal_holdout(ds),
        "offer": offer,
        "disclaimer": (
            "Индивидуальные коэффициенты описывают ассоциации; "
            "ценовая причинность, удержанный объём и uplift не установлены. "
            "Сценарная экономика требует рандомизированного A/B-теста."
        ),
    }
    return summary, scores


def _save_chart(data: list[dict], key: str, ylabel: str, filename: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    ax.plot([r["month"] for r in data], [r[key] for r in data], marker="o", linewidth=2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Месяц")
    ax.grid(alpha=.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_private_scores(path: Path, ds: FuelDataset, scores, policy: OfferPolicy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    promo = eligible_for_offer(ds, scores, policy)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["client_id", "main_station", "region", "fuel", "segment",
                         "beta_per_rub_per_l", "correlation", "loo_negative_count",
                         "residual_price_gap_std", "active_months", "liters_9m",
                         "last_month_liters", "last_month_price_gap", "app", "push",
                         "margin_after_opex_rub_per_l", "eligible_for_pilot"])
        for i, key in enumerate(ds.ids):
            writer.writerow([
                key, ds.stations[i], ds.regions[i], ds.fuels[i], scores.segment[i],
                f"{scores.beta[i]:.6f}", f"{scores.correlation[i]:.6f}",
                int(scores.loo_negative[i]), f"{scores.residual_gap_std[i]:.6f}",
                int(scores.active_months[i]), f"{scores.total_liters[i]:.2f}",
                f"{ds.volume[i, -1]:.2f}", f"{ds.gap[i, -1]:.3f}",
                int(ds.app[i]), int(ds.push[i]),
                f"{ds.contribution_per_liter[i]:.4f}", int(promo[i]),
            ])


def save_results(ds: FuelDataset, output: str | Path, policy: OfferPolicy,
                 private_scores: bool = True) -> dict:
    output = Path(output)
    public = output / "public"
    public.mkdir(parents=True, exist_ok=True)
    summary, scores = build_summary(ds, policy)
    (public / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_chart(summary["monthly"], "liters", "Объём заправок, л", public / "liters.svg")
    _save_chart(summary["monthly"], "mean_price_gap_rub_per_liter",
                "Цена сети − средняя цена конкурентов, руб./л", public / "price_gap.svg")
    if private_scores:
        _write_private_scores(output / "private" / "customer_scores.csv", ds, scores, policy)
    s, a, val = summary["segments"], summary["offer"], summary["validation"]
    f = lambda v: f"{v:,.2f}".replace(",", " ")
    md = f"""# Итоги анализа эластичности спроса

> Данные: {summary['dataset']['period'][0]} — {summary['dataset']['period'][1]}. Исследовательский расчёт; не причинная оценка.

| Метрика | Значение |
|---|---:|
| Клиентов | {summary['dataset']['clients']:,} |
| Основных АЗС | {summary['dataset']['unique_main_stations']:,} |
| Литров за период | {f(summary['dataset']['total_liters'])} |
| Нулевых клиент-месяцев | {summary['dataset']['zero_client_month_share']:.1%} |

## Сегменты

| Сегмент | Клиентов | Доля |
|---|---:|---:|
"""
    for label, x in s.items():
        md += f"| {label} | {x['clients']:,} | {x['share']:.1%} |\n"
    md += "\n## Отложенная проверка (7 месяцев → 2 месяца)\n\n"
    for key, name in (("sensitive", "Чувствительные"), ("stable", "Устойчивые"),
                      ("uncertain", "Неопределённые")):
        data = val["groups"][key]
        b = data["holdout_beta"]
        md += f"- **{name}:** {data['clients']:,} клиентов, β = {b:.3f} руб.⁻¹.\n" if b is not None else f"- **{name}:** невозможно оценить коэффициент.\n"
    md += f"""
## Экономика персонального бонуса (сценарий)

- Получателей: **{a['eligible_clients']:,}**.
- Бонус: **{policy.rebate_per_liter:.2f} руб./л** на первые **{policy.cap_liters:.0f} л**.
- Предполагаемый прирост: **{policy.assumed_uplift:.0%}**.
- Дополнительный объём: **{f(a['incremental_liters'])} л**.
- Полные расходы на бонусы: **{f(a['bonus_cost_rub'])} руб.**
- Инкрементальный вклад: **{f(a['incremental_contribution_rub'])} руб.**
- Чистый дополнительный результат: **{f(a['net_incremental_rub'])} руб.**
- Безубыточный прирост: **{a['breakeven_uplift']:.2%}**.

## Ограничения

{summary['disclaimer']}

Данных о покупках у конкурентов нет. Сегмент «устойчивые» означает отсутствие
устойчивого отрицательного отклика *по принятому правилу*, а не доказанную
нечувствительность к цене. Согласия на коммуникацию нужно перепроверять
непосредственно перед отправкой предложения.
"""
    (public / "summary.md").write_text(md, encoding="utf-8")
    return summary
