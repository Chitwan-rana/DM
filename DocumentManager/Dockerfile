# ========================
# Stage 1: Build
# ========================
FROM python:3.11-slim AS builder

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies (only needed for compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU wheels explicitly
RUN pip install torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install remaining dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ========================
# Stage 2: Runtime
# ========================
FROM python:3.11-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

# Install only runtime dependencies (lighter than build-essential)
RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Pre-download NLTK data
RUN python -m nltk.downloader -d /usr/local/nltk_data \
    punkt \
    averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng \
    stopwords

# Copy project source
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["gunicorn", "DocumentManager.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "120"]
