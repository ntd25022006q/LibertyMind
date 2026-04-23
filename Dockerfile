FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY cli.py .
COPY configs/ configs/

# Install LibertyMind
RUN pip install --no-cache-dir -e ".[all]"

# Expose proxy server port
EXPOSE 8080

# Default: start proxy server
ENTRYPOINT ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8080"]
