# Start from a PyTorch base image with CUDA support
# Check RunPod documentation for recommended CUDA versions for PyTorch or choose a recent one.
# PyTorch 2.1.0 with CUDA 12.1 is a good recent choice.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# ffmpeg is crucial for Whisper to handle various audio formats.
# libsndfile1 is often needed for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Ensure your pip is up-to-date within the build
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and any other necessary files from your repo
# For now, just app.py. If app.py depends on other local modules, copy them too.
COPY app.py .
# If your app.py needs to use code from the 'training' directory or other parts
# of the distil-whisper repo, you would add lines like:
# COPY training /app/training
# COPY other_needed_dirs /app/other_needed_dirs
# For now, the app.py we'll create will be self-contained for inference.

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run your application
# This assumes your app.py will be updated to run a Uvicorn server with FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
