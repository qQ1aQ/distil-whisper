# 1. Use a PyTorch base image with CUDA and Python 3.10
# PyTorch 2.1.0, CUDA 11.8, Python 3.10 (matching torch version from flash-attn error log)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for audio processing
# These might not be in the PyTorch devel image and are needed for libraries like datasets[audio] or torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the /app/ directory
COPY requirements.txt .

# Upgrade pip, then install flash-attn (which requires nvcc and CUDA_HOME from the base image),
# then install other packages from requirements.txt.
# The base image already has PyTorch 2.1.0.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flash-attn --no-build-isolation && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application script into the /app/ directory
COPY app.py .

# Command to run when the container starts
CMD ["python", "app.py"]