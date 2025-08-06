# Dockerfile

FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Expose port (optional)
EXPOSE 8000

# Collect static files (optional)
# RUN python manage.py collectstatic --noinput

# Run the server
CMD ["gunicorn", "documentmanager.wsgi:application", "--bind", "0.0.0.0:8000"]
