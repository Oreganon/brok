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
COPY brok/ ./brok/

# Install the package
RUN pip install --no-cache-dir .

# Runtime stage using distroless
FROM python:3.13-slim as runtime

# Copy the installed packages from builder stage
COPY --from=builder /usr/local /usr/local

# Set working directory
WORKDIR /app

# Make sure scripts in .local are usable and add to Python path
ENV PATH=/usr/local/bin:$PATH

# Default entrypoint uses the brok console script installed by pip
ENTRYPOINT ["brok"] 
