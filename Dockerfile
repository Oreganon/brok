# Build stage
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv pip install --system --no-cache .

# Copy source code
COPY brok/ ./brok/

# Runtime stage
FROM python:3.13-slim as runtime

# Copy the installed packages from builder stage
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY brok/ /app/brok/
COPY main.py /app/

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONPATH=/app

# Default entrypoint
ENTRYPOINT ["python", "main.py"] 
