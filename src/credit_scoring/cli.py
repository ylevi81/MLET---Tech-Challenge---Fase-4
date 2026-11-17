"""Interface de linha de comando para validar lotes, treinar o baseline e monitorar drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from credit_scoring.baseline import split_reference, train_baseline
from credit_scoring.contract import validate_batch
from credit_scoring.data import download_reference, read_batch
from credit_scoring.observability import PipelineRun
from credit_scoring.simulation import simulate_production

SOURCE_XLS = Path("data/credit_default.xls")
PRODUCTION_DIR = Path("artifacts/production")


def execute(args: argparse.Namespace, run: PipelineRun) -> None:
    """Dispatch the parsed command, reporting progress and metrics on ``run``."""

    if args.command == "validate":
        with run.stage("ingestion"):
            batch = validate_batch(read_batch(args.input), labeled=not args.sem_alvo)
            run.fingerprint("input", args.input)
        run.metrics({"pipeline/rows": len(batch)})
        print(f"Lote aceito: {len(batch)} registro(s).")
    elif args.command in ("train", "simulate"):
        with run.stage("ingestion"):
            source = args.input or download_reference(SOURCE_XLS)
            batch = validate_batch(read_batch(source), labeled=True)
            run.fingerprint("input", source)
        run.metrics({"pipeline/input_rows": len(batch)})
        if args.command == "train":
            _run_train(args, run, batch)
        else:
            _run_simulate(args, run, batch)
    else:
        _run_drift(args, run)


def _run_train(args: argparse.Namespace, run: PipelineRun, batch: pd.DataFrame) -> None:
    with run.stage("training"):
        metrics = train_baseline(batch, args.output_dir)
    run.metrics(
        {
            "pipeline/rows": len(batch),
            **{
                f"baseline/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
    )
    run.fingerprint("model", args.output_dir / "baseline.joblib")
    run.artifact(args.output_dir / "metrics.json")
    run.artifact(args.output_dir / "baseline.joblib")
    print(json.dumps(metrics, indent=2))


def _run_simulate(args: argparse.Namespace, run: PipelineRun, batch: pd.DataFrame) -> None:
    with run.stage("transformation"):
        _, holdout = split_reference(batch)
        paths = simulate_production(holdout, args.output_dir)
    run.metrics({"pipeline/rows": len(holdout), "pipeline/batches": len(paths)})
    run.artifact(args.output_dir / "cenario.json")
    for path in paths:
        run.fingerprint(path.stem, path)
        print(f"Lote gravado: {path}")


def _run_drift(args: argparse.Namespace, run: PipelineRun) -> None:
    from credit_scoring.drift import monitor

    with run.stage("drift"):
        run.fingerprint("model", args.model)
        run.fingerprint("reference", args.reference)
        for path in sorted(args.production_dir.glob("mes_*.csv")):
            run.fingerprint(path.stem, path)
        summary = monitor(args.reference, args.production_dir, args.model, args.report_dir)
        run.drift(summary, args.report_dir)
    print_monitoring(summary, args.report_dir)


def print_monitoring(summary: dict[str, Any], report_dir: Path) -> None:
    """Print one line per simulated month plus the location of the Evidently report."""

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


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI and convert expected operational failures into exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        with PipelineRun(args.command) as run:
            execute(args, run)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
