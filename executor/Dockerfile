FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (optimized for size)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install core ML libraries
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    lightgbm==4.1.0 \
    xgboost==2.0.3 \
    catboost==1.2.2 \
    flaml==2.1.1

# Install transformers and NLP libraries
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    tokenizers==0.14.1 \
    sentencepiece==0.1.99

# Install image processing libraries
RUN pip install --no-cache-dir \
    albumentations==1.3.1 \
    Pillow==10.1.0

# Install audio processing libraries
RUN pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    audioread==3.0.1

# Install utilities
RUN pip install --no-cache-dir \
    tqdm==4.66.1 \
    joblib==1.3.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0

# Create workspace directory
RUN mkdir -p /app/workspace

# Set working directory to workspace
WORKDIR /app/workspace

# Default command (will be overridden)
CMD ["python", "generated_code.py"]
