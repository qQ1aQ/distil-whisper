# Use the specified PyTorch base image with CUDA 12.1 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies
# - ffmpeg and libsndfile1 are for audio processing
# - git is useful for some pip package installations that might clone repos, or for versioning within the container if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# --upgrade pip ensures the latest pip is used
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This assumes your app.py and any other necessary code (like a 'models' directory if local)
# are in the same directory as the Dockerfile.
COPY app.py .
# If you have other directories or files needed by app.py, copy them here. For example:
# COPY ./your_model_directory /app/your_model_directory

# Expose the port FastAPI will run on (if you're using FastAPI in app.py for serving)
# Default for uvicorn is 8000. Adjust if your app.py uses a different port.
EXPOSE 8000

# Command to run when the container starts
# This will execute your app.py script using python.
# If app.py uses uvicorn to serve a FastAPI app, the command would be different, e.g.:
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]_
# For now, keeping the original assumption for app.py:
CMD ["python", "app.py"]
