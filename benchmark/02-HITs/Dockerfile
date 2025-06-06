# Use an official Python runtime as a parent image
# FROM python:3.10-slim
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Re-declare build args after FROM for use in this stage
ARG UID
ARG GID
ARG DIR=HierarchyTransformers

# Create a group and user matching host UID/GID
RUN groupadd -g ${GID} usergroup \
    && useradd -m -u ${UID} -g usergroup user

# Set the working directory
WORKDIR /workspace

# Copy repository contents
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r ${DIR}/requirements.txt

# Prepare unit_test results directory and set permissions
RUN mkdir -p /workspace/${DIR}/unit_test/logs \
    && chown -R user:usergroup /workspace/${DIR}/unit_test/logs

# Run as non-root user to match host permissions
USER user

# # Expose results directory as a volume
# VOLUME ["/workspace/${DIR}/unit_test/results"]