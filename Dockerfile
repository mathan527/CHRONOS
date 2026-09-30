FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHRONOS_TRACE_DIR=/data/traces \
    CHRONOS_DB=/data/chronos.db \
    CHRONOS_WEB_DIR=/app/web \
    CHRONOS_SAMPLES_DIR=/app/scenarios/images

RUN useradd --create-home --uid 1000 chronos && mkdir -p /data/traces && chown -R chronos /data
WORKDIR /app

# Dependencies first so code edits don't invalidate this layer.
COPY pyproject.toml ./
RUN mkdir chronos && touch chronos/__init__.py \
    && pip install . && pip uninstall -y chronos && rm -rf chronos build

COPY chronos ./chronos
COPY web ./web
COPY scenarios ./scenarios
RUN pip install --no-deps . && chown -R chronos /app

USER chronos
EXPOSE 8000
VOLUME /data
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "chronos.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
