ARG UID
ARG GID
ARG DIR=direct-preference-optimization

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Re-declare build args after FROM for use in this stage
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

# Install Python dependencies directly
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        einops \
        local-attention \
        matplotlib \
        numpy \
        pandas \
        patool \
        reformer-pytorch \
        scikit-learn \
        scipy \
        sktime \
        sympy \
        tqdm \
        psutil \
        peft==0.10.0

# Prepare unit_test results directory and set permissions
RUN mkdir -p /workspace/${DIR}/unit_test/logs \
    && chown -R user:usergroup /workspace/${DIR}/unit_test/logs

# Run as non-root user to match host permissions
USER user
