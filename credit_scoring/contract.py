"""Contrato de entrada para dados de referência e lotes de pontuação."""

import pandas as pd
import pandera.pandas as pa


CUSTOMER_ID = "customer_id"
TARGET = "default_next_month"
SOURCE_FEATURES = {
    "AGE": "age",
    "LIMIT_BAL": "credit_limit",
    "PAY_0": "payment_status_m1",
    "PAY_2": "payment_status_m2",
    "PAY_3": "payment_status_m3",
    "BILL_AMT1": "bill_amount_m1",
    "BILL_AMT2": "bill_amount_m2",
    "BILL_AMT3": "bill_amount_m3",
    "PAY_AMT1": "payment_amount_m1",
    "PAY_AMT2": "payment_amount_m2",
    "PAY_AMT3": "payment_amount_m3",
}
FEATURES = list(SOURCE_FEATURES.values())


def validate_batch(frame: pd.DataFrame, *, labeled: bool) -> pd.DataFrame:
    """Rejeita o lote inteiro e reúne as violações em uma única exceção."""
    columns = {
        CUSTOMER_ID: pa.Column(int, pa.Check.gt(0), unique=True),
        "age": pa.Column(int, pa.Check.in_range(19, 120)),
        "credit_limit": pa.Column(int, pa.Check.gt(0)),
    }
    for month in range(1, 4):
        columns[f"payment_status_m{month}"] = pa.Column(int, pa.Check.in_range(-2, 9))
        columns[f"bill_amount_m{month}"] = pa.Column(int)
        columns[f"payment_amount_m{month}"] = pa.Column(int, pa.Check.ge(0))
    if labeled:
        columns[TARGET] = pa.Column(int, pa.Check.isin([0, 1]))

    schema = pa.DataFrameSchema(
        columns,
        checks=pa.Check(lambda batch: not batch.empty, name="lote_nao_vazio"),
        strict=True,
        unique_column_names=True,
        name="credit_batch",
    )
    return schema.validate(frame, lazy=True)
