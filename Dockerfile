# --- Builder Stage ---
FROM python:3.11-bookworm AS builder
WORKDIR /app
COPY requirements.txt .
# Install to a specific prefix to make copying easier
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runner Stage ---
FROM python:3.11-slim-bookworm AS runner

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed packages from the builder
COPY --from=builder /install /usr/local
# Copy your application code
COPY . .

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Ensure NLTK data is stored in a predictable place
ENV NLTK_DATA=/app/nltk_data

# Pre-download NLTK data into the image
RUN python -m nltk.downloader -d /app/nltk_data punkt punkt_tab

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]