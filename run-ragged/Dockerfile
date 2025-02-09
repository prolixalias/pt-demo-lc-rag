# ----------------------- FRONTEND BUILD STAGE -----------------------
  FROM node:20 AS frontend-builder
  WORKDIR /frontend
  
  COPY frontend/package*.json ./
  RUN npm install -g npm@11.1.0
  RUN npm install
  COPY frontend/src ./src
  COPY frontend/public ./public
  COPY frontend/index.html .
  COPY frontend/vite.config.js .
  COPY frontend/tailwind.config.js .
  COPY frontend/postcss.config.js .
  RUN npm run build || (cat /frontend/npm-debug.log 2>/dev/null || true && exit 1)
  
  
  # ----------------------- SPHINX BUILD STAGE -----------------------
  FROM python:3.11-slim-bullseye AS sphinx-builder
  WORKDIR /docs

  COPY . /code/
  COPY docs .
  RUN pip install --upgrade pip
  RUN pip install sphinx sphinx-rtd-theme httpx fastapi python-json-logger prometheus_client json-repair langchain-core google-cloud-storage langchain_community langchain-google-vertexai cloud-sql-python-connector langchain langchain-xai opentelemetry-api opentelemetry-instrumentation circuitbreaker pypdf
  RUN PYTHONPATH=/code sphinx-build -b html . _build/html
  
  
  # ----------------------- PYTHON BUILDER STAGE -----------------------
  FROM python:3.11-slim-bullseye AS builder
  WORKDIR /code

  COPY --from=sphinx-builder /docs/_build/html /code/frontend/dist/docs
  
  ENV PYTHONUNBUFFERED=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="${PYTHONPATH}:/" \
      POETRY_VERSION=1.8.5 \
      POETRY_HOME="/opt/poetry" \
      POETRY_VIRTUALENVS_IN_PROJECT=true \
      POETRY_NO_INTERACTION=1 \
      VIRTUAL_ENV="/code/.venv"
  
  RUN python -m venv $VIRTUAL_ENV
  RUN pip install --upgrade pip
  
  ENV PATH="$VIRTUAL_ENV/bin:$PATH"
  RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"
  
  COPY pyproject.toml poetry.lock README.md ./
  RUN poetry lock --no-update
  RUN poetry install --no-root --no-cache --no-interaction --no-ansi
  
  
  # ----------------------- FINAL RUNTIME IMAGE -----------------------
  FROM python:3.11-slim-bullseye
  WORKDIR /code

  COPY --from=builder /code/.venv /code/.venv
  ENV PATH="/code/.venv/bin:$PATH"
  
  RUN apt-get update \
      && apt-get install -y --no-install-recommends -o APT::Immediate-Configure=false \
        libpq5 libpq-dev gcc g++ curl \
      && rm -rf /var/lib/apt/lists/*
  
  COPY ./app /code/app
  COPY pyproject.toml poetry.lock README.md ./
  
  COPY --from=frontend-builder /frontend/dist /code/frontend/dist
  COPY --from=frontend-builder /frontend/public/assets /code/frontend/public/assets
  COPY --from=sphinx-builder /docs/_build/html /code/frontend/dist/docs
    
  RUN poetry lock --no-update
  RUN poetry install --no-interaction --no-ansi
  RUN adduser --disabled-password --gecos "" appuser \
      && chown -R appuser:appuser /code
  USER appuser
  
  EXPOSE 8080
  HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3   CMD curl -f http://localhost:8080/health || exit 1
  
  CMD uvicorn app.$SERVICE_TYPE:app --host 0.0.0.0 --port 8080