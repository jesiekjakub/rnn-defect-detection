# Multi-stage build: builder resolves and installs Python deps; runtime ships
# only the venv and source. Kept separate so editing app code doesn't bust the
# dep-install layer.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY dashboard ./dashboard
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid 1000 --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8000 \
    MODELS_DIR=/app/models

RUN mkdir -p /app/models/cache && chown -R app:app /app/models

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "dashboard.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
