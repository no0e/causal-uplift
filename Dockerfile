# The whole pipeline in one CPU image.
#
#   docker build -t causal-uplift .
#   docker run --rm causal-uplift                              # synthetic smoke run
#   docker run --rm -v $PWD/data:/app/data -v $PWD/docs:/app/docs causal-uplift \
#       sh -c "python scripts/fetch_criteo.py && python scripts/run_pipeline.py"
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md ./
COPY causal_uplift causal_uplift
RUN pip install --no-cache-dir .
COPY scripts scripts

RUN useradd --create-home runner && mkdir -p data docs && chown -R runner /app
USER runner
CMD ["sh", "-c", "python scripts/fetch_criteo.py --synthetic 200000 && \
     python scripts/run_pipeline.py --data data/synthetic.parquet --quick --out /tmp/docs"]
