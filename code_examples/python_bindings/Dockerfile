# Multi-stage build for smaller final image
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    pybind11[global] \
    numpy \
    setuptools \
    wheel

WORKDIR /app

# Copy source files
COPY . .

# Build the C++ extension
RUN python3 setup.py build_ext --inplace

# Final stage
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

WORKDIR /app

# Copy built extension and Python files
COPY --from=builder /app/*.so .
COPY --from=builder /app/python ./python

# Run benchmark by default
CMD ["python3", "python/benchmark.py"]