FROM python:3.11-slim

# 安装系统依赖
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# 设置环境变量
ENV PYTHONPATH=/app

# 默认运行server.py
CMD ["python", "server.py"]