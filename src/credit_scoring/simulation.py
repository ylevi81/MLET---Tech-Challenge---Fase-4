"""Lotes mensais de produção com inflação, novo perfil de clientes e mudança de conceito."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.contract import CUSTOMER_ID, TARGET, validate_batch

BILL_COLUMNS = [f"bill_amount_m{month}" for month in range(1, 4)]
LIMIT_STEP = 10_000


@dataclass(frozen=True)
class Scenario:
    """Choque aplicado por inteiro no último mês e de forma linear nos anteriores."""

    months: int = 6
    seed: int = 42
    bill_inflation: float = 0.8
    newcomer_share: float = 0.5
    newcomer_age: tuple[int, int] = (19, 29)
    newcomer_limit_factor: float = 0.5
    stress_utilization: float = 0.7
    stress_default_rate: float = 0.3


def drift_batch(
    batch: pd.DataFrame, intensity: float, scenario: Scenario, rng: np.random.Generator
) -> pd.DataFrame:
    """Com intensidade zero o lote sai inalterado e serve de controle."""
    drifted = batch.copy()

    # Inflação: as faturas sobem em valor nominal e os pagamentos não acompanham.
    inflated = drifted[BILL_COLUMNS] * (1 + scenario.bill_inflation * intensity)
    drifted[BILL_COLUMNS] = inflated.round().astype("int64")

    # Novo perfil: clientes mais jovens entram com limites menores.
    newcomer = rng.random(len(drifted)) < scenario.newcomer_share * intensity
    youngest, oldest = scenario.newcomer_age
    drifted.loc[newcomer, "age"] = rng.integers(youngest, oldest + 1, newcomer.sum())
    reduced = drifted.loc[newcomer, "credit_limit"] * scenario.newcomer_limit_factor
    steps = (reduced // LIMIT_STEP).clip(lower=1).astype("int64")
    drifted.loc[newcomer, "credit_limit"] = steps * LIMIT_STEP

    # Conceito: com a renda apertada, quem compromete o limite passa a inadimplir mais.
    utilization = drifted["bill_amount_m1"] / drifted["credit_limit"]
    stressed = (drifted[TARGET] == 0) & (utilization >= scenario.stress_utilization)
    defaulted = stressed & (rng.random(len(drifted)) < scenario.stress_default_rate * intensity)
    drifted.loc[defaulted, TARGET] = 1
    return drifted


def simulate_production(
    holdout: pd.DataFrame, output_dir: Path, scenario: Scenario = Scenario()
) -> list[Path]:
    """Divide a avaliação da etapa 1 em meses e só grava se todos passarem no contrato."""
    rng = np.random.default_rng(scenario.seed)
    shuffled = holdout.sample(frac=1, random_state=scenario.seed)
    intensities = np.linspace(0, 1, scenario.months)
    positions = np.array_split(np.arange(len(shuffled)), scenario.months)

    months = [
        validate_batch(drift_batch(shuffled.iloc[rows], intensity, scenario, rng), labeled=True)
        for rows, intensity in zip(positions, intensities, strict=True)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, month in enumerate(months, start=1):
        path = output_dir / f"mes_{number:02d}.csv"
        month.sort_values(CUSTOMER_ID).to_csv(path, index=False)
        paths.append(path)
    lineage = asdict(scenario) | {"intensity_by_month": [round(float(i), 2) for i in intensities]}
    (output_dir / "cenario.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    return paths
