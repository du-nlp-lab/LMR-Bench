# Use an official Python runtime as a parent image
# FROM python:3.10-slim
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel


# Re-declare build args after FROM for use in this stage
ARG UID
ARG GID
ARG DIR

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libaio-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a group and user matching host UID/GID
RUN groupadd -g ${GID} usergroup \
    && useradd -m -u ${UID} -g usergroup user



# Set proper CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create cache directories with proper permissions
RUN mkdir -p /tmp/cache && \
    mkdir -p /home/user/.triton/autotune && \
    chown -R user:usergroup /tmp/cache && \
    chown -R user:usergroup /home/user/.triton && \
    chmod -R 777 /tmp/cache && \
    chmod -R 777 /home/user/.triton

# Set the TRANSFORMERS_CACHE environment variable to a writable directory
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache

# Run as non-root user to match host permissions
USER user


# Set the working directory
WORKDIR /workspace


# Copy repository contents
COPY --chown=user:usergroup . .



RUN pip install  --no-cache-dir \
    accelerate==0.29.2 \
    bitsandbytes==0.43.0 \
    black==24.4.2 \
    datasets==2.18.0 \
    deepspeed==0.14.4 \
    einops==0.6.1 \
    evaluate==0.4.0 \
    flake8==6.0.0 \
    hf-doc-builder==0.4.0 \
    hf_transfer==0.1.4 \
    huggingface-hub==0.23.0 \
    isort==5.12.0 \
    ninja==1.11.1 \
    numpy==1.24.2 \
    packaging==23.0 \
    parameterized==0.9.0 \
    peft==0.9.0 \
    protobuf==3.20.2 \
    pytest \
    safetensors==0.4.1 \
    sentencepiece==0.1.99 \
    scipy \
    tensorboard \
    torch==2.1.2 \
    transformers==4.39.3 \
    trl==0.9.6 \
    jinja2==3.0.0 \
    tqdm==4.64.1



