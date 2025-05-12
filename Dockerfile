# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies (if any, for audio processing etc.)
# You might need to add more depending on the specific Whisper dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
# Assuming requirements.txt is in the root of your forked repo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary code from the repository
# Copy the app.py entry point script and potentially other needed files/directories
# If your app.py needs other files from the repo, copy them here.
# For this basic example, we'll just copy app.py
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

# If you need to copy other directories from the original repo for your script:
# COPY training/flax /app/training/flax
# COPY other_needed_dirs /app/other_needed_dirs

# Command to run when the container starts
# This now runs your app.py script
CMD ["python", "app.py"]
