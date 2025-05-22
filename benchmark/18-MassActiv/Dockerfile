# 使用 Python 3.11 + CUDA 11.8 的官方 PyTorch 镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel


ARG UID
ARG GID
ARG DIR=MassActiv-main

# 安装依赖
RUN apt-get update && apt-get install -y git

# 添加用户
RUN groupadd -g ${GID} usergroup && useradd -m -u ${UID} -g usergroup user

WORKDIR /workspace
COPY . .

# 设置 HF 的本地缓存路径，确保你已经通过 -v 映射到了这个目录
ENV HF_HOME=/workspace/${DIR}/unit_test/.cache
ENV TRANSFORMERS_OFFLINE=1


# 安装 Python 依赖
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r ${DIR}/requirements.txt

RUN mkdir -p /workspace/${DIR}/unit_test/logs \
    && chown -R user:usergroup /workspace/${DIR}/unit_test/logs

USER user
