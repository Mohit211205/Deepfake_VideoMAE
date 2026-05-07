FROM python:3.10-slim

# System dependencies needed for OpenCV, PIL, and ffmpeg
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (Docker cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# HF Spaces runs as user 1000, set permissions
RUN chmod -R 777 /app

# Expose HF Spaces default port
EXPOSE 7860

# Start FastAPI on port 7860
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
