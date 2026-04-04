# ============================================================
# SkySim Docker image — multi-stage build
# Stage 1: Build the React frontend
# Stage 2: Python runtime with FastAPI serving the built UI
# ============================================================

# --- Stage 1: Build frontend ---
FROM node:20-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY web/ ./
RUN npm run build

# --- Stage 2: Python runtime ---
FROM python:3.11-slim AS runtime
WORKDIR /app

# System dependencies for healpy (needs cfitsio) and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ pkg-config libcfitsio-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "jax[cpu]>=0.4.20" \
    "numpy>=1.24" \
    "astropy>=5.3" \
    "healpy>=1.16" \
    "fastapi>=0.104" \
    "uvicorn[standard]>=0.24" \
    "pillow>=10.0"

# Copy Python package
COPY skysim/ ./skysim/
RUN pip install --no-cache-dir -e .

# Copy built frontend from stage 1
COPY --from=frontend /app/web/dist ./web/dist

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Run server
CMD ["uvicorn", "skysim.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
