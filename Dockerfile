# Use an official NVIDIA CUDA runtime image as a base
# Make sure the CUDA version in the image matches your PyTorch build (cu118 for PyTorch 2.1.0+cu118)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables to non-interactive (prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python, pip, ffmpeg (for audio), and other common dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    software-properties-common \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python3 and pip point to python3.10's pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip, setuptools, and wheel, and install packaging (needed for flash-attn build)
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir packaging

# Set the working directory in the container
WORKDIR /app

# --- Pip Cache Purge and Clean PyTorch Installation ---
# Purge pip cache first
RUN pip cache purge

# Install PyTorch, torchvision, and torchaudio first with --no-cache-dir and --force-reinstall
# This explicitly targets the CUDA 11.8 compatible versions.
RUN pip install \
    --no-cache-dir \
    --force-reinstall \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy the requirements file and install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# If your app.py is at the root of your GitHub repo:
COPY app.py .
# If you have other directories/files your app.py needs, copy them here too
# e.g., COPY my_utils/ /app/my_utils/

# Expose the port the app runs on
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
