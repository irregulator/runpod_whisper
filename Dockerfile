# Use NVIDIA CUDA base image with cuDNN (required for GPU Whisper)
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Update and install system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash libgl1 libx11-6 \
        software-properties-common ffmpeg build-essential python3.10 python3.10-dev python3.10-venv python3-pip && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


# Upgrade pip and install Whisper, transformers, and dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install openai-whisper transformers requests


# (Optional) Install runpod SDK if using serverless handler
RUN pip install runpod

# Copy your handler file (replace with your actual handler)
COPY rp_handler.py /

# Set default command for Runpod serverless worker
CMD ["python3", "-u", "rp_handler.py"]
