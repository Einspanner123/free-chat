# Reuse the pinned fork image's native binaries; all Python changes originate on ross.
ARG WORKER_BASE=freechat-worker:8e78a3c61307
FROM ${WORKER_BASE}
COPY third_party/vllm/vllm/v1/engine/core.py /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py
COPY libs/contracts/src /opt/freechat/libs/contracts/src
COPY libs/protocol/src /opt/freechat/libs/protocol/src
COPY worker/src /opt/freechat/worker/src
COPY tools/validate_execution_gpu.py tools/validate_native_http.py /opt/freechat/tools/
RUN uv venv /opt/freechat/.venv --system-site-packages --python /usr/bin/python3
WORKDIR /opt/freechat
ENV PYTHONPATH=/opt/freechat:/opt/freechat/libs/contracts/src:/opt/freechat/libs/protocol/src:/opt/freechat/worker/src
ENTRYPOINT ["/opt/freechat/.venv/bin/python", "-m", "freechat_worker.serve"]
