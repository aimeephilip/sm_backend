FROM python:3.11-slim

# Install system libraries that OpenCV/MediaPipe need
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Make a working directory
WORKDIR /app

# Prevent Python from writing .pyc files etc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy your list of Python dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your whole project into the container
COPY . .

# Tell Render how to start your app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
