import os


class AppConfig:
    def __init__(self):
        # 应用配置
        self.serverName = os.getenv("SERVER_NAME", "llm-inference")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.grpcPort = int(os.getenv("GRPC_PORT", 8083))

        # 模型配置
        self.modelName = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
        self.maxTokens = int(os.getenv("MAX_TOKENS", 100))
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))

        # 系统配置
        self.maxWorkers = int(os.getenv("MAX_WORKERS", 10))


config = AppConfig()
