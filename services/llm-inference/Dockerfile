FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 安装Python 3.12
RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# 复制项目文件
COPY cmd/llm-inference/ .

# 安装uv并使用它安装依赖
RUN pip install uv && \
    uv venv && \
    uv pip install --no-cache-dir -r pyproject.toml && \
    # 预编译Python字节码
    uv run python -m compileall src/

# 设置环境变量
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD uv run python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); \
    grpc.channel_ready_future(channel).result(timeout=1)" || exit 1

CMD ["uv", "run", "src/server.py"]