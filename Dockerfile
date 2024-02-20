# Use a CUDA 11.8 base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /usr/src/app

# Set noninteractive to avoid interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="8.6"

# Update and install system packages, then clean up APT cache
RUN apt-get update && \
    apt-get -y install --no-install-recommends build-essential git software-properties-common libssl-dev&& \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3-pip python3-venv python3.10-dev python3-distutils python3-apt&& \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Ensure pip, setuptools, and wheel are up to date
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN which nvcc && nvcc --version

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/julien-piet/cpp-hash.git && \
    cd cpp-hash && \
    python3 setup.py install
# Copy the requirements.txt file into the container
COPY requirements.txt ./requirements.txt
# Install Python dependencies
RUN pip install -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install evaluate