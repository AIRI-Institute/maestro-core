# Dockerfile for mmar-ptag
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY tests/ ./tests/

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies
RUN uv pip install --system -e .

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]
