"""Interface para validar lotes, treinar o baseline e monitorar drift na produção."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

from pandera.errors import SchemaError, SchemaErrors

from .baseline import split_reference, train_baseline
from .contract import validate_batch
from .data import download_reference, read_batch
from .simulation import simulate_production


SOURCE_XLS = Path("data/credit_default.xls")
PRODUCTION_DIR = Path("artifacts/production")


def print_monitoring(summary: dict, report_dir: Path) -> None:
    for month in summary["months"]:
        drifted = ", ".join(month["drifted_features"]) or "nenhuma"
        status = "ALERTA" if month["drifted_features"] or month["performance_alert"] else "ok"
        print(
            f"{month['batch']} [{status}] drift em: {drifted} | "
            f"inadimplência real {month['default_rate']:.1%} | "
            f"ROC AUC {month['roc_auc']:.3f} | recall {month['default_recall']:.1%}"
            + (" | desempenho abaixo do baseline" if month["performance_alert"] else "")
        )
    print(f"Relatório Evidently de {summary['report_batch']}: {report_dir / 'drift_report.html'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validação, baseline e monitoramento de drift para risco de crédito."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Valida um lote sem treinar o modelo.")
    validate.add_argument("input", type=Path, help="Caminho do CSV ou XLS original da UCI.")
    validate.add_argument("--sem-alvo", action="store_true", help="Valida um CSV para pontuação.")

    train = commands.add_parser("train", help="Valida e treina o modelo de referência.")
    train.add_argument("--input", type=Path, help="CSV normalizado ou XLS original da UCI.")
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))

    simulate = commands.add_parser("simulate", help="Gera lotes mensais de produção com drift.")
    simulate.add_argument("--input", type=Path, help="CSV normalizado ou XLS original da UCI.")
    simulate.add_argument("--output-dir", type=Path, default=PRODUCTION_DIR)

    drift = commands.add_parser("drift", help="Pontua os lotes e compara com a referência.")
    drift.add_argument("--reference", type=Path, default=Path("artifacts/reference.csv"))
    drift.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    drift.add_argument("--model", type=Path, default=Path("artifacts/baseline.joblib"))
    drift.add_argument("--report-dir", type=Path, default=Path("reports"))

    args = parser.parse_args()
    try:
        if args.command == "validate":
            batch = validate_batch(read_batch(args.input), labeled=not args.sem_alvo)
            print(f"Lote aceito: {len(batch)} registro(s).")
        elif args.command == "train":
            source = args.input or download_reference(SOURCE_XLS)
            metrics = train_baseline(read_batch(source), args.output_dir)
            print(json.dumps(metrics, indent=2))
        elif args.command == "simulate":
            source = args.input or download_reference(SOURCE_XLS)
            _, holdout = split_reference(validate_batch(read_batch(source), labeled=True))
            for path in simulate_production(holdout, args.output_dir):
                print(f"Lote gravado: {path}")
        else:
            # O import tardio evita carregar o Evidently nos outros comandos.
            from .drift import monitor

            summary = monitor(args.reference, args.production_dir, args.model, args.report_dir)
            print_monitoring(summary, args.report_dir)
    except (SchemaError, SchemaErrors) as error:
        print("Lote rejeitado pelo contrato. Ingestão bloqueada.", file=sys.stderr)
        summary = error.failure_cases.groupby(["column", "check"], dropna=False, sort=False).size()
        for (column, check), count in summary.items():
            print(f"- {column}: {check} ({count} ocorrência(s))", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        print(f"Falha ao processar o lote: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
