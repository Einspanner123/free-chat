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
        # Engine type: "vllm" (default), "hf", "auto"
        self.engineType = os.getenv("ENGINE_TYPE", "vllm")
        # Quantization: None (FP16), "awq", "gptq", "squeezellm"
        self.quantization: Optional[str] = os.getenv("QUANTIZATION") or None
        # vLLM-specific settings
        self.gpuMemoryUtilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
        self.tensorParallelSize = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        self.maxModelLen = int(os.getenv("MAX_MODEL_LEN", "8192"))

        # ---- Speculative Decoding Configuration ----
        # DRAFT_MODEL: HF model id of the draft model. None disables speculative decoding.
        self.draftModel: Optional[str] = os.getenv("DRAFT_MODEL") or None
        # SPECULATIVE_GAMMA: number of draft tokens proposed per verify round.
        self.speculativeGamma = int(os.getenv("SPECULATIVE_GAMMA", "5"))
        # SPECULATIVE_ENABLED: secondary gate; draft_model_path is the master opt-out.
        self.speculativeEnabled = os.getenv("SPECULATIVE_ENABLED", "true").lower() in (
            "1", "true", "yes",
        )

        # ---- KV Cache Eviction (StreamingLLM-style) Configuration ----
        # KV_EVICTION_WINDOW: recent tokens kept per layer. 0 disables eviction.
        # KV_EVICTION_SINK: attention-sink tokens always kept (recommend >= 4).
        self.kvEvictionSink = int(os.getenv("KV_EVICTION_SINK", "4"))
        kv_window = os.getenv("KV_EVICTION_WINDOW", "0")
        self.kvEvictionWindow: Optional[int] = int(kv_window) or None

        # ---- Prefix KV Cache (serve-time reuse) Configuration ----
        # PREFIX_CACHE_ENABLED: reuse prefilled KV for shared prompt prefixes
        # (system prompt / RAG context) to skip re-prefill. Default off.
        self.prefixCacheEnabled = os.getenv("PREFIX_CACHE_ENABLED", "false").lower() in (
            "1", "true", "yes",
        )
        self.prefixCacheCapacity = int(os.getenv("PREFIX_CACHE_CAPACITY", "8"))

        # ---- KV Compression (MLA-style latent, opt-in) Configuration ----
        # KV_COMPRESSION: "none" (default) or "mla"
        self.kvCompression = os.getenv("KV_COMPRESSION", "none")
        self.kvCompressionLatent = int(os.getenv("KV_COMPRESSION_LATENT", "16"))
        self.kvCompressionBasis = os.getenv("KV_COMPRESSION_BASIS") or None

        # 系统配置
        self.maxWorkers = int(os.getenv("MAX_WORKERS", 10))


config = AppConfig()
