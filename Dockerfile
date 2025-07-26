# Build stage
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for better layer caching
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir --user .

# Runtime stage using distroless
FROM gcr.io/distroless/python3-debian12:latest

# Copy the installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY main.py /app/

# Set working directory
WORKDIR /app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Set the entrypoint
ENTRYPOINT ["python", "main.py"] 