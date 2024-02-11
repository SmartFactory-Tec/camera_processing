ARG CUDA=11.8.0
ARG CUDNN=8
ARG OPENCV=4.7.0
ARG UBUNTU_VERSION=22.04


FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU_VERSION} as runner
ARG CUDA
ARG CUDNN
ARG OPENCV
ARG BUILD_CORES=2

# Specify custom configuration folder, to be overriden via volume
ENV CAMERA_PROCESSING_CONFIG_FOLDER /config

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Install all dependencies to build OpenCV with CUDA support
# noninteractive required to skip installation prompts
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-numpy \
    python3 \
    pipenv \
    wget \
    unzip

# Copy sems-processors project
WORKDIR /camera_processing

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

# Install dependencies to system python
RUN pipenv install --categories "packages cuda-packages system-packages" --system --deploy --ignore-pipfile
RUN pip install --no-deps ultralytics

COPY camera_processing camera_processing
COPY models models

# Run gunicorn
ENTRYPOINT ["python", "-m", "camera_processing"]
