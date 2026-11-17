"""Execuções, eventos estruturados e recursos do processo no MLflow."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any

import mlflow
import psutil
from pandera.errors import SchemaError, SchemaErrors


class PipelineRun:
    """Tracks one CLI command as an MLflow run with structured JSONL events."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.root = Path(os.environ.get("CREDIT_MONITORING_DIR", "monitoring")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        default_uri = f"sqlite:///{self.root / 'mlflow.db'}"
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
        experiment = "credit-scoring"
        if mlflow.get_experiment_by_name(experiment) is None:
            # Local runs need no server; HTTP runs use the server's artifact destination.
            local = mlflow.get_tracking_uri().startswith("sqlite:")
            artifact_location = (self.root / "artifacts").as_uri() if local else None
            mlflow.create_experiment(experiment, artifact_location=artifact_location)
        mlflow.set_experiment(experiment)
        self.samples: list[dict[str, float]] = []
        self.stop = Event()

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "command": self.command,
            "event": event,
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def sample(self) -> None:
        process = psutil.Process()
        cpu = process.cpu_times()
        self.samples.append(
            {
                "infra/process_rss_mb": process.memory_info().rss / 1024**2,
                "infra/process_cpu_seconds": cpu.user + cpu.system,
                "infra/host_memory_percent": psutil.virtual_memory().percent,
                "infra/disk_free_gb": psutil.disk_usage(str(self.root)).free / 1024**3,
            }
        )

    def sampling(self) -> None:
        while not self.stop.wait(1):
            try:
                self.sample()
            except psutil.Error:
                self.event("resource_sampling_failed")

    def __enter__(self) -> PipelineRun:
        self.active = mlflow.start_run(run_name=self.command)
        self.run_id = self.active.info.run_id
        self.log_path = self.root / "logs" / f"{self.run_id}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tags({"pipeline.command": self.command, "pipeline.outcome": "running"})
        self.started = perf_counter()
        self.event("run_started")
        try:
            self.sample()
        except Exception:
            mlflow.end_run(status="FAILED")
            raise
        self.worker = Thread(target=self.sampling, daemon=True)
        self.worker.start()
        return self

    def metrics(self, values: dict[str, float], step: int = 0) -> None:
        mlflow.log_metrics(values, step=step)

    def artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def fingerprint(self, name: str, path: Path) -> None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        mlflow.log_param(f"{name}_sha256", digest.hexdigest())

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        self.event("stage_started", stage=name)
        try:
            yield
        except BaseException as error:
            self.event("stage_failed", stage=name, error_type=type(error).__name__)
            self.metrics({f"stage/{name}/success": 0})
            raise
        else:
            self.event("stage_finished", stage=name)
            self.metrics({f"stage/{name}/success": 1})
        finally:
            self.metrics({f"stage/{name}/duration_seconds": perf_counter() - started})

    def drift(self, summary: dict[str, Any], report_dir: Path) -> None:
        mlflow.log_params(summary["rules"])
        alerts = 0
        for step, month in enumerate(summary["months"], 1):
            drift_alert = bool(month["drifted_features"]) or month["score_drift"]
            alerts += int(drift_alert) + int(month["performance_alert"])
            self.metrics(
                {
                    "model/roc_auc": month["roc_auc"],
                    "model/default_recall": month["default_recall"],
                    "model/default_rate": month["default_rate"],
                    "model/predicted_default_rate": month["predicted_default_rate"],
                    "drift/features": len(month["drifted_features"]),
                    "alerts/drift": int(drift_alert),
                    "alerts/performance": int(month["performance_alert"]),
                    "batch/rows": month["rows"],
                },
                step,
            )
            self.event("drift_result", **month)
        self.metrics(
            {"pipeline/rows": sum(m["rows"] for m in summary["months"]), "pipeline/alerts": alerts}
        )
        for filename in ("drift_summary.json", "drift_timeline.csv", "drift_report.html"):
            self.artifact(report_dir / filename)

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: Any,
    ) -> bool:
        self.stop.set()
        self.worker.join()
        rejected = isinstance(error, (SchemaError, SchemaErrors))
        outcome = "success" if error is None else "rejected" if rejected else "failed"
        try:
            self.sample()
            # Exception messages and source rows can contain personal data; retain only type.
            self.event(
                "run_finished",
                outcome=outcome,
                error_type=error_type.__name__ if error_type else None,
            )
            for step, sample in enumerate(self.samples):
                self.metrics(sample, step)
            self.metrics(
                {
                    "pipeline/duration_seconds": perf_counter() - self.started,
                    "pipeline/success": int(error is None),
                    "alerts/failure": int(error is not None),
                    "alerts/contract": int(rejected),
                    "infra/peak_sampled_rss_mb": max(
                        s["infra/process_rss_mb"] for s in self.samples
                    ),
                }
            )
            mlflow.set_tag("pipeline.outcome", outcome)
            self.artifact(self.log_path)
        except BaseException:
            outcome = "failed"
            raise
        finally:
            mlflow.end_run(status="FINISHED" if outcome == "success" else "FAILED")
        return False
