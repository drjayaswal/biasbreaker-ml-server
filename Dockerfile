FROM python:3.11-slim-bookworm

# 1. Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the Hugging Face User
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/home/user/app/nltk_data \
    HF_HOME=/home/user/app/.cache

WORKDIR $HOME/app

# 3. Install Python packages in CHUNKS
# Breaking these up prevents the builder from timing out or hitting RAM limits
COPY --chown=user:user requirements.txt .

# Chunk 1: Install Torch CPU specifically (Smallest footprint)
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Chunk 2: Everything else from requirements.txt
# (Pip will skip Torch since it's already there)
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app code
COPY --chown=user:user . .

# 5. Download NLTK data using a python call (more reliable than downloader script)
RUN python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.1', 'stopwords'], download_dir='/home/user/app/nltk_data')"

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]