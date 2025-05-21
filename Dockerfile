# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
#testtest comment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir packaging

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# If you have other directories/files your app.py needs, copy them here too

EXPOSE 8000
CMD ["python3", "app.py"] 
# Still diagnostic app.py