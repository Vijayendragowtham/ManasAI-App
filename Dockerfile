FROM python:3.11-slim

# Audio libraries needed by librosa / soundfile (wav, mp3, ogg, m4a)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# XSRF protection is turned off because file uploads fail with a 403 error
# inside the Hugging Face Spaces iframe when it is on.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
