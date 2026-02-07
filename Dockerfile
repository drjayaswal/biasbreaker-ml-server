FROM python:3.11-slim-bookworm

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/home/user/app/nltk_data \
    HF_HOME=/home/user/app/.cache

WORKDIR $HOME/app

# 3. Install dependencies
# We install Torch CPU first to ensure we don't pull a 2GB GPU version
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY --chown=user:user . .

# 5. Pre-download NLTK data
RUN python -m nltk.downloader -d $HOME/app/nltk_data punkt punkt_tab averaged_perceptron_tagger_eng wordnet omw-1.1 stopwords

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]