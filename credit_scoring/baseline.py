"""Treinamento e avaliação da regressão logística de referência."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contract import CUSTOMER_ID, FEATURES, TARGET, validate_batch


RANDOM_STATE = 42
MIN_SAMPLES = 5_000
HOLDOUT_SHARE = 0.2
THRESHOLD = 0.5


def split_reference(validated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa a referência de treino da avaliação, que a etapa 2 usa como base de produção."""
    # A ordenação torna a divisão reproduzível mesmo se o CSV vier em outra ordem.
    ordered = validated.sort_values(CUSTOMER_ID)
    reference, holdout = train_test_split(
        ordered,
        test_size=HOLDOUT_SHARE,
        random_state=RANDOM_STATE,
        stratify=ordered[TARGET],
    )
    return reference, holdout


def train_baseline(batch: pd.DataFrame, output_dir: Path) -> dict:
    validated = validate_batch(batch, labeled=True)
    if len(validated) < MIN_SAMPLES:
        raise ValueError(f"O treinamento exige pelo menos {MIN_SAMPLES} registros válidos.")
    class_counts = validated[TARGET].value_counts()
    if class_counts.size != 2 or class_counts.min() < 5:
        raise ValueError("O treinamento exige pelo menos cinco registros de cada classe.")

    reference, holdout = split_reference(validated)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, random_state=RANDOM_STATE),
    )
    model.fit(reference[FEATURES], reference[TARGET])

    actual = holdout[TARGET]
    scores = model.predict_proba(holdout[FEATURES])[:, 1]
    predicted = (scores >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    metrics = {
        "seed": RANDOM_STATE,
        "threshold": THRESHOLD,
        "reference_rows": len(reference),
        "holdout_rows": len(holdout),
        "reference_default_rate": float(reference[TARGET].mean()),
        "holdout_default_rate": float(actual.mean()),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "default_precision": float(precision_score(actual, predicted, zero_division=0)),
        "default_recall": float(recall_score(actual, predicted, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    reference[[CUSTOMER_ID, *FEATURES, TARGET]].to_csv(output_dir / "reference.csv", index=False)
    joblib.dump(model, output_dir / "baseline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
