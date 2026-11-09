"""CLI для анализа оригинального файла и безопасного синтетического демо."""
import argparse
from pathlib import Path

from .demo import make_demo_dataset
from .economics import OfferPolicy
from .io import load_xlsb
from .reporting import save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fuel-elasticity", description="Intelligence Cup: ценовая чувствительность АЗС")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Обработка приватного файла кейса .xlsb")
    run.add_argument("--input", required=True, type=Path, help="Путь к исходному .xlsb")
    demo = sub.add_parser("demo", help="Демонстрация на синтетических данных")
    demo.add_argument("--clients", type=int, default=1200)
    demo.add_argument("--seed", type=int, default=42)
    for command in (run, demo):
        command.add_argument("--output", type=Path, default=Path("outputs"))
        command.add_argument("--rebate", type=float, default=.5, help="Бонус, руб/л")
        command.add_argument("--cap", type=float, default=70, help="Лимит, л")
        command.add_argument("--uplift", type=float, default=.05, help="Прирост 0.05 = 5%%")
        command.add_argument("--price-gap", type=float, default=.5)
        command.add_argument("--public-only", action="store_true",
                             help="Не создавать закрытый CSV с клиентскими данными")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = OfferPolicy(args.rebate, args.cap, args.uplift, args.price_gap)
    policy.validate()
    if args.command == "run":
        ds = load_xlsb(args.input)
    else:
        ds = make_demo_dataset(n=args.clients, seed=args.seed)
    summary = save_results(ds, args.output, policy, private_scores=not args.public_only)
    offer = summary["offer"]
    print(f"Клиентов: {summary['dataset']['clients']:,}")
    print("Сегменты:", ", ".join(f"{k}: {v['clients']:,}"
                                     for k, v in summary["segments"].items()))
    print(f"Пилот: {offer['eligible_clients']:,} человек")
    print(f"Сценарный чистый эффект: {offer['net_incremental_rub']:,.2f} руб.")
    print(f"Агрегированные отчёты: {args.output / 'public'}")
    if not args.public_only:
        print(f"ЗАКРЫТЫЕ результаты (не публиковать): {args.output / 'private'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
