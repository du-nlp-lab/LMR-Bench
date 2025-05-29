# Base image with PyTorch, CUDA, and cuDNN runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Re-declare build args for use in this stage
ARG UID
ARG GID
ARG DIR


# Create a matching host user & group
RUN groupadd -g ${GID} usergroup \
 && useradd -m -u ${UID} -g usergroup user

# Set working directory
WORKDIR /workspace

# Copy the entire repo
# COPY . .

# Install Python requirements
# RUN pip install --upgrade pip setuptools wheel \
#  && pip install --no-cache-dir -r ${DIR}/requirements.txt

# Initialize and update any git submodules
# RUN git submodule update --init --recursive


# Install the package in editable mode
RUN pip install \
        torch>=2.5 \
        transformers>=4.45.0 \
        triton>=3.0 \
        datasets>=3.3.0 \
        einops \
        ninja \
        pytest

# Prepare unit test logs and fix ownership
RUN mkdir -p /workspace/${DIR}/unit_test/logs \
 && chown -R user:usergroup /workspace/${DIR}/unit_test/logs

# Switch to non-root user for all further operations
USER user
