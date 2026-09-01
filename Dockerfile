# twin-turbofan — reproducible core pipeline.
#
# Deliberately installs only requirements.txt (numpy/pandas/matplotlib/sklearn), which
# keeps the image near 500MB. Torch is ~2.5GB and is not needed for the RandomForest
# baseline, the feature ablation, or the test suite — the sequence-model tests skip
# themselves when torch is absent. Add it explicitly if you want the sequence models:
#
#   docker build --build-arg WITH_TORCH=1 -t twin-turbofan .
#
# The real C-MAPSS data is not baked in (licence + size). Mount it at run time:
#
#   docker run --rm -v "$PWD/data/CMAPSSData:/app/data/CMAPSSData" twin-turbofan
#
# With no data mounted, the entrypoint generates the synthetic fallback so the image
# still demonstrates the pipeline end to end.

FROM python:3.11-slim

ARG WITH_TORCH=0

WORKDIR /app

# Matplotlib needs no display; fail loudly rather than trying to open one.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$WITH_TORCH" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY src/ ./src/
COPY tests/ ./tests/
COPY conftest.py pytest.ini pyproject.toml ./
COPY data/README.md ./data/

# Generate synthetic data only if the real set was not mounted, then run the pipeline.
CMD ["sh", "-c", "\
    if [ ! -f data/CMAPSSData/train_FD001.txt ]; then \
        echo '>> no real data mounted; generating synthetic fallback'; \
        python -m src.generate_synthetic; \
    fi && \
    python -m pytest -q && \
    python -m src.train_baseline && \
    python -m src.error_analysis"]
