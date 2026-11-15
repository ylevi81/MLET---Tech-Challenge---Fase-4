FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_ENABLE_TELEMETRY=false DO_NOT_TRACK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && useradd --create-home --uid 10001 app \
    && mkdir /state && chown app:app /state /app
COPY --chown=app:app credit_scoring ./credit_scoring
COPY --chown=app:app examples ./examples
USER app
ENV GIT_PYTHON_REFRESH=quiet MLFLOW_DISABLE_AGENT_HINT=1
CMD ["python", "-m", "credit_scoring", "--help"]
