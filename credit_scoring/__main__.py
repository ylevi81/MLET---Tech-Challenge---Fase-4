"""Interface para validar lotes e reproduzir o treinamento da etapa 1."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

from pandera.errors import SchemaError, SchemaErrors

from .baseline import train_baseline
from .contract import validate_batch
from .data import download_reference, read_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Validação e baseline para risco de crédito.")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Valida um lote sem treinar o modelo.")
    validate.add_argument("input", type=Path, help="Caminho do CSV ou XLS original da UCI.")
    validate.add_argument("--sem-alvo", action="store_true", help="Valida um CSV para pontuação.")

    train = commands.add_parser("train", help="Valida e treina o modelo de referência.")
    train.add_argument("--input", type=Path, help="CSV normalizado ou XLS original da UCI.")
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))

    args = parser.parse_args()
    try:
        if args.command == "validate":
            batch = validate_batch(read_batch(args.input), labeled=not args.sem_alvo)
            print(f"Lote aceito: {len(batch)} registro(s).")
        else:
            source = args.input or download_reference(Path("data/credit_default.xls"))
            metrics = train_baseline(read_batch(source), args.output_dir)
            print(json.dumps(metrics, indent=2))
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
