FROM python:3.11-slim

WORKDIR /app

COPY cmd/llm-inference/ .

RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 运行应用
CMD ["python", "hello.py"]
