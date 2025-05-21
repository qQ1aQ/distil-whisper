# Step 1: Start from an official PyTorch image with corresponding CUDA and Python versions.
# This image comes with PyTorch 2.1.0, CUDA 11.8, Python 3.10, and cuDNN 8 pre-installed.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Step 2: Set environment variables (optional, but good practice)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Step 3: Install any *additional* system dependencies your application needs.
# The PyTorch base image is quite comprehensive (includes git, build-essential, etc.),
# but we previously identified ffmpeg and libsndfile1 for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    # Add any other essential system packages here if needed
    && rm -rf /var/lib/apt/lists/*

# Step 4: Upgrade pip and install 'packaging' (which was a build dependency for flash-attn).
# We use the pip from the PyTorch image's environment (usually conda-based).
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir packaging

# Step 5: Set the working directory in the container
WORKDIR /app

# Step 6: Copy your requirements.txt file.
# IMPORTANT: It's highly recommended to REMOVE or COMMENT OUT torch, torchvision, and torchaudio
# from your requirements.txt file for this step, so we use the versions from the base image.
COPY requirements.txt .

# Step 7: Install Python dependencies from requirements.txt.
# Since PyTorch is already in the base image, pip should skip it if it's not in your
# (modified) requirements.txt, or confirm it's satisfied.
RUN pip install --no-cache-dir -r requirements.txt

# Step 8: Copy the rest of your application code.
# For now, this is still our diagnostic app.py.
COPY app.py .
# If you have other directories/files your app.py needs, copy them here too
# e.g., COPY my_utils/ /app/my_utils/

# Step 9: Expose the port the app runs on (for when we switch to the full app)
EXPOSE 8000

# Step 10: Command to run when the container starts.
# This will run your diagnostic app.py first.
# The PyTorch base image usually sets up PATH correctly so 'python' or 'python3'
# will point to the correct Python interpreter within its pre-configured environment.
CMD ["python3", "app.py"]
# When ready for the full app, this will change to:
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
