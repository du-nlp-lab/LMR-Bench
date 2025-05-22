FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Re-declare build args after FROM for use in this stage
ARG UID
ARG GID
ARG DIR=TimeDistill

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
