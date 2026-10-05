# =============================================================================
# Dockerfile - Symptom-to-Disease-to-Specialist Prediction API
# =============================================================================
FROM python:3.12-slim

# System environment variables for Python in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install curl for healthcheck probe
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create a secure non-root system user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy and install only runtime dependencies to maximize layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what is strictly needed for runtime inference
COPY src/ /app/src/
COPY config/ /app/config/
COPY data/disease_specialist.csv /app/data/disease_specialist.csv
COPY models/production/ /app/models/production/

# Ensure runtime directories exist with non-root ownership
RUN mkdir -p /app/data/feedback /app/data/prod_logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Container healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
