# Build stage
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code for installation
COPY pyproject.toml .
COPY main.py .
COPY brok/ ./brok/

# Install the package
RUN pip install --no-cache-dir --user .

# Runtime stage using distroless
FROM gcr.io/distroless/python3-debian12:latest

# Copy the installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set the entrypoint
ENTRYPOINT ["brok"] 
