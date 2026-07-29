import os
from typing import Optional


class AppConfig:
    def __init__(self):
        # 应用配置
        self.serverName = os.getenv("SERVER_NAME", "llm-inference")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.grpcPort = int(os.getenv("GRPC_PORT", 8083))

        # 模型配置
        self.modelName = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
        self.maxTokens = int(os.getenv("MAX_TOKENS", 512))
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))
        self.repetitionPenalty = float(os.getenv("REPETITION_PENALTY", 1.1))
        self.topP = float(os.getenv("TOP_P", 0.8))
        self.topK = int(os.getenv("TOP_K", 40))

        # ---- New: Engine & Quantization Configuration ----
        # Engine type: "auto", "vllm", "hf"
        self.engineType = os.getenv("ENGINE_TYPE", "auto")
        # Quantization: None (FP16), "awq", "gptq", "squeezellm"
        self.quantization: Optional[str] = os.getenv("QUANTIZATION") or None
        # vLLM-specific settings
        self.gpuMemoryUtilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
        self.tensorParallelSize = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        self.maxModelLen = int(os.getenv("MAX_MODEL_LEN", "8192"))

        # 系统配置
        self.maxWorkers = int(os.getenv("MAX_WORKERS", 10))


config = AppConfig()
