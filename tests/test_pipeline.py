import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from pandera.errors import SchemaErrors

from credit_scoring.baseline import split_reference, train_baseline
from credit_scoring.contract import TARGET, validate_batch
from credit_scoring.drift import monitor, performance, SCORE, PREDICTION
from credit_scoring.observability import PipelineRun
from credit_scoring.simulation import simulate_production


def valid_batch(rows=5000):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"customer_id": np.arange(1, rows + 1),
                          "age": rng.integers(19, 70, rows),
                          "credit_limit": rng.integers(1, 20, rows) * 10000})
    for month in range(1, 4):
        frame[f"payment_status_m{month}"] = rng.integers(-2, 5, rows)
        frame[f"bill_amount_m{month}"] = rng.integers(0, 150000, rows)
        frame[f"payment_amount_m{month}"] = rng.integers(0, 10000, rows)
    frame[TARGET] = (frame.payment_status_m1 > 1).astype("int64")
    return frame


class PipelineTests(unittest.TestCase):
    def test_contract_rejects_bad_data(self):
        for column, value in (("age", 18), ("credit_limit", 0), ("payment_amount_m1", -1), ("age", None)):
            with self.subTest(column=column, value=value):
                frame = valid_batch(10)
                frame.loc[0, column] = value
                with self.assertRaises(SchemaErrors):
                    validate_batch(frame, labeled=True)
        frame = valid_batch(10)
        frame.loc[0, "customer_id"] = 2
        with self.assertRaises(SchemaErrors):
            validate_batch(frame, labeled=True)

    def test_invalid_training_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            with self.assertRaises(SchemaErrors):
                train_baseline(pd.read_csv("examples/lote_invalido.csv"), output)
            self.assertFalse(output.exists())

    def test_split_and_simulation(self):
        frame = valid_batch()
        reference, holdout = split_reference(frame)
        reordered, _ = split_reference(frame.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(reference, reordered)
        self.assertFalse(set(reference.customer_id) & set(holdout.customer_id))
        with tempfile.TemporaryDirectory() as folder:
            paths = simulate_production(holdout, Path(folder))
            months = [pd.read_csv(path) for path in paths]
            self.assertEqual(sum(map(len, months)), len(holdout))
            original = holdout.set_index("customer_id").loc[months[0].customer_id].reset_index()
            pd.testing.assert_frame_equal(months[0], original)
            self.assertFalse(set(reference.customer_id) & set(pd.concat(months).customer_id))

    def test_drift_validates_every_batch_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            valid_batch(10).to_csv(root / "reference.csv", index=False)
            valid_batch(10).to_csv(root / "mes_01.csv", index=False)
            broken = valid_batch(10)
            broken.loc[0, "age"] = 10
            broken.to_csv(root / "mes_02.csv", index=False)
            with self.assertRaises(SchemaErrors):
                monitor(root / "reference.csv", root, root / "absent.joblib", root / "reports")
            self.assertFalse((root / "reports").exists())
            self.assertFalse((root / "predicoes.csv").exists())

    def test_single_class_is_not_reported_as_healthy(self):
        with self.assertRaisesRegex(ValueError, "duas classes"):
            performance(pd.DataFrame({TARGET: [0, 0], SCORE: [.1, .2], PREDICTION: [0, 0]}))

    def test_overlapping_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            valid_batch(10).to_csv(root / "reference.csv", index=False)
            valid_batch(10).to_csv(root / "mes_01.csv", index=False)
            with self.assertRaisesRegex(ValueError, "clientes distintos"):
                monitor(root / "reference.csv", root, root / "absent.joblib", root / "reports")
            self.assertFalse((root / "reports").exists())

    def test_tracking_success_rejection_and_unexpected_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = {"CREDIT_MONITORING_DIR": folder, "MLFLOW_TRACKING_URI": f"sqlite:///{root / 'mlflow.db'}"}
            with patch.dict(os.environ, env):
                with PipelineRun("validate") as success:
                    with success.stage("ingestion"):
                        success.metrics({"pipeline/rows": 10})
                with self.assertRaises(SchemaErrors):
                    with PipelineRun("validate") as rejected:
                        with rejected.stage("ingestion"):
                            validate_batch(pd.read_csv("examples/lote_invalido.csv"), labeled=True)
                with self.assertRaises(RuntimeError):
                    with PipelineRun("simulate") as failed:
                        raise RuntimeError("private-customer-value")
                client = MlflowClient()
                for tracked, outcome in ((success, "success"), (rejected, "rejected"), (failed, "failed")):
                    run = client.get_run(tracked.run_id)
                    self.assertEqual(run.data.tags["pipeline.outcome"], outcome)
                    self.assertEqual(run.info.status, "FINISHED" if outcome == "success" else "FAILED")
                    self.assertGreater(run.data.metrics["infra/peak_sampled_rss_mb"], 0)
                    self.assertTrue(client.list_artifacts(tracked.run_id))
                    events = [json.loads(line) for line in tracked.log_path.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(events[-1]["outcome"], outcome)
                    self.assertNotIn("private-customer-value", tracked.log_path.read_text())
                self.assertEqual(client.get_run(rejected.run_id).data.metrics["alerts/contract"], 1)
            # Release SQLite pools on Windows before TemporaryDirectory cleanup.
            from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
            for engine in SqlAlchemyStore._engine_map.values():
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
