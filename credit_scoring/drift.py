"""Predições, testes PSI e Kolmogorov-Smirnov e relatório Evidently dos lotes de produção."""

import json
from pathlib import Path

import joblib
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .baseline import THRESHOLD
from .contract import CUSTOMER_ID, FEATURES, TARGET, validate_batch


SCORE = "default_score"
PREDICTION = "predicted_default"
MONITORED = [*FEATURES, SCORE]
# PSI de 0,1 a 0,25 indica mudança moderada e acima de 0,25, forte.
PSI_LIMIT = 0.1
PSI_STRONG = 0.25
KS_ALPHA = 0.05
# O Evidently marca o lote com metade das colunas alteradas; em crédito, 30% já pede revisão.
DATASET_DRIFT_SHARE = 0.3
AUC_TOLERANCE = 0.03
RECALL_TOLERANCE = 0.10


def predict(model, batch: pd.DataFrame) -> pd.DataFrame:
    scored = batch.copy()
    scored[SCORE] = model.predict_proba(batch[FEATURES])[:, 1]
    scored[PREDICTION] = (scored[SCORE] >= THRESHOLD).astype(int)
    return scored


def as_dataset(scored: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=MONITORED,
        categorical_columns=[TARGET, PREDICTION],
        classification=[
            BinaryClassification(
                target=TARGET, prediction_labels=PREDICTION, prediction_probas=SCORE
            )
        ],
    )
    return Dataset.from_pandas(scored[[*MONITORED, TARGET, PREDICTION]], data_definition=definition)


def column_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Uma coluna só é marcada quando PSI e KS concordam, o que reduz falsos alarmes."""
    tests = [ValueDrift(column=column, method=method) for column in MONITORED for method in ("psi", "ks")]
    snapshot = Report(tests).run(as_dataset(current), as_dataset(reference))
    values: dict[str, dict[str, float]] = {}
    for result in snapshot.dict()["metrics"]:
        config = result["config"]
        values.setdefault(config["column"], {})[config["method"]] = float(result["value"])

    table = pd.DataFrame.from_dict(values, orient="index").loc[MONITORED]
    table = table.rename(columns={"ks": "ks_pvalue"}).rename_axis("column").reset_index()
    table["drift"] = (table["psi"] >= PSI_LIMIT) & (table["ks_pvalue"] < KS_ALPHA)
    table["severity"] = "estavel"
    table.loc[table["drift"], "severity"] = "moderado"
    table.loc[table["drift"] & (table["psi"] >= PSI_STRONG), "severity"] = "forte"
    return table


def performance(scored: pd.DataFrame) -> dict:
    actual, predicted = scored[TARGET], scored[PREDICTION]
    return {
        "rows": len(scored),
        "default_rate": float(actual.mean()),
        "predicted_default_rate": float(predicted.mean()),
        "roc_auc": float(roc_auc_score(actual, scored[SCORE])),
        "default_recall": float(recall_score(actual, predicted, zero_division=0)),
        "default_precision": float(precision_score(actual, predicted, zero_division=0)),
    }


def monitor(
    reference_path: Path, production_dir: Path, model_path: Path, report_dir: Path
) -> dict:
    """Valida todos os lotes antes de pontuar e grava o resumo, a série mensal e o HTML."""
    batch_paths = sorted(production_dir.glob("mes_*.csv"))
    if not batch_paths:
        raise FileNotFoundError(f"Nenhum lote mes_*.csv em {production_dir}. Rode simulate antes.")
    reference_batch = validate_batch(pd.read_csv(reference_path), labeled=True)
    batches = {path.stem: validate_batch(pd.read_csv(path), labeled=True) for path in batch_paths}

    model = joblib.load(model_path)
    baseline = json.loads(model_path.with_name("metrics.json").read_text(encoding="utf-8"))
    reference = predict(model, reference_batch)
    scored = {name: predict(model, batch) for name, batch in batches.items()}

    tables, months = [], []
    for name, batch in scored.items():
        table = column_drift(reference, batch)
        tables.append(table.assign(batch=name))
        quality = performance(batch)
        drifted = table.loc[table["drift"] & table["column"].isin(FEATURES), "column"].tolist()
        months.append(
            {
                "batch": name,
                **quality,
                "drifted_features": drifted,
                "score_drift": bool(table.loc[table["column"] == SCORE, "drift"].item()),
                "performance_alert": (
                    quality["roc_auc"] < baseline["roc_auc"] - AUC_TOLERANCE
                    or quality["default_recall"] < baseline["default_recall"] - RECALL_TOLERANCE
                ),
            }
        )

    latest_name, latest = next(reversed(scored.items()))
    report = Report(
        [
            DataDriftPreset(
                columns=MONITORED, method="psi", threshold=PSI_LIMIT, drift_share=DATASET_DRIFT_SHARE
            ),
            DataDriftPreset(
                columns=MONITORED, method="ks", threshold=KS_ALPHA, drift_share=DATASET_DRIFT_SHARE
            ),
            ClassificationPreset(),
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report.run(as_dataset(latest), as_dataset(reference)).save_html(str(report_dir / "drift_report.html"))

    timeline = pd.concat(tables, ignore_index=True)
    timeline[["batch", "column", "psi", "ks_pvalue", "drift", "severity"]].to_csv(
        report_dir / "drift_timeline.csv", index=False, float_format="%.6g"
    )
    predictions = pd.concat(
        frame[[CUSTOMER_ID, SCORE, PREDICTION, TARGET]].assign(batch=name)
        for name, frame in scored.items()
    )
    predictions.to_csv(production_dir / "predicoes.csv", index=False)

    summary = {
        "reference_rows": len(reference),
        "report_batch": latest_name,
        "rules": {
            "psi_limit": PSI_LIMIT,
            "psi_strong": PSI_STRONG,
            "ks_alpha": KS_ALPHA,
            "dataset_drift_share": DATASET_DRIFT_SHARE,
            "baseline_roc_auc": baseline["roc_auc"],
            "baseline_default_recall": baseline["default_recall"],
            "auc_tolerance": AUC_TOLERANCE,
            "recall_tolerance": RECALL_TOLERANCE,
        },
        "months": months,
    }
    (report_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
