# Frontend build stage
FROM node:20 AS frontend-builder
WORKDIR /frontend

# Copy package files for better caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the frontend source code
# Make sure your local structure matches this
COPY frontend/src ./src
COPY frontend/public ./public
COPY frontend/index.html .
COPY frontend/vite.config.js .
COPY frontend/tailwind.config.js .
COPY frontend/postcss.config.js .

# Build the frontend
RUN npm run build || (cat /frontend/npm-debug.log 2>/dev/null || true && exit 1)

# Python dependencies stage
FROM python:3.11-slim-bullseye AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${PYTHONPATH}:/" \
    POETRY_VERSION=1.8.5 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    VIRTUAL_ENV="/code/.venv"

# Create a virtual environment for the project
WORKDIR /code
RUN python -m venv $VIRTUAL_ENV

# Install Poetry inside the virtual environment to reduce image size
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy only necessary files for Poetry to install dependencies
COPY pyproject.toml poetry.lock README.md ./

# Install dependencies with the updated pyproject.toml
RUN poetry lock --no-update
RUN poetry install --no-root --no-cache --no-interaction --no-ansi

# Final runtime image
FROM python:3.11-slim-bullseye

# Copy over the virtual environment from the builder stage
COPY --from=builder /code/.venv /code/.venv
ENV PATH="/code/.venv/bin:$PATH"

# Set work directory
WORKDIR /code

# Install runtime dependencies only
RUN apt-get update \
    && apt-get install -y --no-install-recommends -o APT::Immediate-Configure=false \
      libpq5 libpq-dev gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY ./app /code/app
COPY pyproject.toml poetry.lock README.md ./

# Copy built React files from frontend-builder stage
COPY --from=frontend-builder /frontend/dist /code/frontend/dist
COPY --from=frontend-builder /frontend/public/assets /code/frontend/public/assets

# Install the application itself
RUN poetry lock --no-update
RUN poetry install --no-interaction --no-ansi

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /code

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD uvicorn app.$SERVICE_TYPE:app --host 0.0.0.0 --port 8080
