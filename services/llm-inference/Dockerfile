FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 更换 apt 源为阿里云镜像
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装Python 3.12 (通过源码编译，避开PPA网络问题)
RUN apt-get update && \
    apt-get install -y wget build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
    libffi-dev uuid-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://npmmirror.com/mirrors/python/3.12.8/Python-3.12.8.tar.xz && \
    tar -xf Python-3.12.8.tar.xz && \
    cd Python-3.12.8 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf Python-3.12.8 Python-3.12.8.tar.xz

# 创建软链接
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.12 /usr/bin/python3

WORKDIR /app

# 复制项目文件
COPY services/llm-inference/ .

# 安装uv并使用它安装依赖
RUN python -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    uv venv && \
    uv pip install --no-cache-dir -r pyproject.toml && \
    # 预编译Python字节码
    uv run python -m compileall src/

# 设置环境变量
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD uv run python -c "import grpc; channel = grpc.insecure_channel('localhost:8083'); \
    grpc.channel_ready_future(channel).result(timeout=1)" || exit 1

CMD ["uv", "run", "src/server.py"]