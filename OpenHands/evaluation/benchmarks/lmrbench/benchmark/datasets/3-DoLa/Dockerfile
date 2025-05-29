ARG DIR=DoLa-main
ARG UID
ARG GID

# Use an official Python runtime as a parent image
# FROM python:3.10-slim
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Re-declare build args after FROM for use in this stage
ARG UID
ARG GID
ARG DIR

ENV DIR=${DIR}


# Set the working directory in the container
WORKDIR /workspace

# Create a group and user matching host UID/GID
RUN groupadd -g ${GID} usergroup \
    && useradd -m -u ${UID} -g usergroup user

# Copy the folder into the container
COPY . .


RUN pip install --upgrade pip setuptools wheel

RUN pip install -r ${DIR}/requirements.txt --no-deps

RUN pip install "tokenizers>=0.11.1,!=0.11.3,<0.14" --no-deps


# Prepare unit_test results directory and set permissions
RUN mkdir -p /workspace/${DIR}/unit_test/logs \
    && chown -R user:usergroup /workspace/${DIR}/unit_test/logs


# Run as non-root user to match host permissions
USER user



