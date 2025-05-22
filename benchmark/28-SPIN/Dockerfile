# Use the official PyTorch image with development tools
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     CUDA_HOME=/usr/local/cuda


ARG UID
ARG GID
ARG DIR

# Create a group and user matching host UID/GID
RUN groupadd -g ${GID} usergroup \
    && useradd -m -u ${UID} -g usergroup user


# Set the working directory
WORKDIR /workspace

# Copy repository contents
COPY . .

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     CUDA_HOME=/usr/local/cuda

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     ca-certificates \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     ninja-build \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies as specified in setup.py
RUN pip install \
    vllm==0.3.0 \
    ray==2.9.0 \
    accelerate==0.23.0 \
    bitsandbytes==0.41.2.post2 \
    black==23.1.0 \
    datasets==2.14.6 \
    deepspeed==0.12.2 \
    einops>=0.6.1 \
    evaluate==0.4.0 \
    flake8>=6.0.0 \
    hf-doc-builder>=0.4.0 \
    "huggingface-hub>=0.14.1,<1.0" \
    isort>=5.12.0 \
    ninja>=1.11.1 \
    numpy>=1.24.2 \
    packaging>=23.0 \
    parameterized>=0.9.0 \
    peft==0.6.1 \
    "protobuf<=3.20.2" \
    pytest \
    safetensors>=0.3.3 \
    scipy \
    tensorboard \
    torch>=2.1.0 \
    transformers==4.37.0 \
    trl==0.7.4 \
    "jinja2>=3.0.0" \
    "tqdm>=4.64.1"

# # Clone the SPIN repository
# WORKDIR /app
# RUN git clone https://github.com/uclaml/SPIN.git .

# Install the SPIN package
# RUN python -m pip install .

# Install flash-attn
RUN pip install packaging ninja
# RUN python -m pip install flash-attn --no-build-isolation

RUN mkdir -p /workspace/${DIR}/unit_test/logs \
    && chown -R user:usergroup /workspace/${DIR}/unit_test/logs

USER user
