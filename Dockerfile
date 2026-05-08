# FROM python:3.13-slim

# WORKDIR /app

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# RUN pip install --no-cache-dir uv

# COPY pyproject.toml uv.lock ./
# RUN uv venv /app/.venv && \
#     . /app/.venv/bin/activate && \
#     uv sync --frozen --no-dev

# COPY src/ src/
# COPY evals/ evals/
# COPY data/ data/

# ENV PATH="/app/.venv/bin:$PATH" \
#     PORT=8080

# CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]

FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency metadata first for better layer caching
COPY pyproject.toml uv.lock ./

# Create virtualenv and install locked dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen --no-dev

# Copy application source and eval runtime assets
COPY src/ src/
COPY evals/ evals/

ENV PATH="/app/.venv/bin:$PATH"

ENV PORT=8080

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port $PORT"]
