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
ENV SEMS_CONFIG_FOLDER /config

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

# Download and extract OpenCV sources
WORKDIR /
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPENCV}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV}.zip && \
    unzip opencv.zip -d / && \
    unzip opencv_contrib.zip -d /

# Build and install OpenCV
WORKDIR /opencv-${OPENCV}/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
     	-D CMAKE_INSTALL_PREFIX=/usr/local \
    # fix for a bug in installation path of cv2 bindings
        -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
     	-D INSTALL_PYTHON_EXAMPLES=OFF \
     	-D INSTALL_C_EXAMPLES=OFF \
     	-D OPENCV_ENABLE_NONFREE=ON \
     	-D WITH_CUDA=ON \
     	-D WITH_CUDNN=ON \
     	-D OPENCV_DNN_CUDA=ON \
     	-D ENABLE_FAST_MATH=1 \
     	-D CUDA_FAST_MATH=1 \
     	-D CUDA_ARCH_BIN=7.5 \
     	-D WITH_CUBLAS=1 \
     	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV}/modules \
     	-D HAVE_opencv_python3=ON \
    ..
RUN make -j${BUILD_CORES}
RUN make install && ldconfig

# Remove sources from final image
RUN rm -rf /opencv-${OPENCV}/build
RUN rm -rf /opencv_contrib-${OPENCV}/modules

# Copy sems-vision project
WORKDIR /sems-vision

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

# Install dependencies to system python
RUN pipenv install --system --deploy --ignore-pipfile

# Use gunicorn as the WSGI server
RUN pip install gunicorn

COPY sems_vision sems_vision
COPY models models
COPY static static
COPY templates templates

# Run gunicorn
ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0:3000","sems_vision:create_app()"]
