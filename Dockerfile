FROM python:3.9-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE="/code/cache" \
    TORCH_HOME="/code/torch" \
    PYTHONPATH="/code" \
    PORT=8080

# Create cache directories
RUN mkdir -p /code/cache /code/torch /code/nltk_data

# Set NLTK_DATA path
ENV NLTK_DATA=/code/nltk_data

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data with error handling
RUN python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy the rest of the application
COPY . .

# Set permissions
RUN chmod -R 755 /code

# Expose port
EXPOSE 8080

# Command to run the application with optimized settings
CMD ["gunicorn", "main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "300", \
     "--worker-tmp-dir", "/dev/shm", \
     "--log-level", "info", \
     "--access-logfile", "-"]
