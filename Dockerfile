# 1. Use a smaller NVIDIA CUDA base image (CUDA 11.8, Ubuntu 22.04)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Set the working directory
WORKDIR /app

# 2. Install Python, pip, git, and essential build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    git \
    build-essential \
    software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make python${PYTHON_VERSION} the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# 3. Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    rm -rf ~/.cache/pip

# 4. Install PyTorch 2.1.0 for CUDA 11.8
# This version combination is important for flash-attn and model compatibility.
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    rm -rf ~/.cache/pip

# 5. Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 6. Copy requirements file
COPY requirements.txt .

# 7. Install build dependencies 'setuptools', 'wheel', 'packaging', then a specific version of flash-attn and other packages from requirements.txt
# Pinning flash-attn to 2.5.8 for better compatibility with PyTorch 2.1.0.
# Let pip handle build isolation for flash-attn.
RUN python3 -m pip install --no-cache-dir --upgrade setuptools wheel packaging && \
    python3 -m pip install --no-cache-dir flash-attn==2.5.8 && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# 8. Copy application script
COPY app.py .

# Command to run when the container starts
CMD ["python3", "app.py"]