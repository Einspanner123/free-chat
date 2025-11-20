import signal
import sys
from concurrent import futures
from typing import Iterator

import grpc
import llm_inference_pb2 as pb2
import llm_inference_pb2_grpc as pb2_grpc
from chat_model import ChatModel
from config import config
from loguru import logger


# 实现 gRPC 服务类
class InferencerServiceServicer(pb2_grpc.InferencerServiceServicer):
    def __init__(self):
        # 这里可以初始化模型和其他资源
        logger.info(f"初始化服务，使用模型: {config.modelName}")
        self.model = ChatModel()

    def StreamInference(
        self, request_iterator: Iterator[pb2.InferenceRequest], context
    ) -> Iterator[pb2.InferenceResponse]:
        """实现流式推理方法"""
        logger.info("接收到流式推理请求")

        try:
            # 收集请求信息
            session_id = None
            messages = []
            temperature = config.temperature

            for request in request_iterator:
                session_id = request.session_id
                if request.message:
                    messages.append(request.message)
                if request.temperature > 0:
                    temperature = request.temperature
                logger.info(
                    f"接收到消息: session_id={session_id}, message_length={len(request.message)}, temperature={temperature}"
                )

            streamer = self.model.GetStreamer(msg=messages)
            gen_tokens = 0
            try:
                for chunk in streamer:
                    gen_tokens += len(self.model.tokenizer.tokenize(chunk))

                    yield pb2.InferenceResponse(
                        chunk=chunk,
                        is_finished=False,
                        error="",
                        generated_tokens=gen_tokens,
                    )
            except Exception as e:
                logger.error(f"流式推理过程中出错: {str(e)}")
                yield pb2.InferenceResponse(
                    chunk="",
                    is_finished=True,
                    error=str(e),
                    generated_tokens=gen_tokens,
                )
            # 发送结束信号
            yield pb2.InferenceResponse(
                chunk="",
                is_finished=True,
                error="",
                generated_tokens=gen_tokens,
            )

        except Exception as e:
            logger.error(f"流式推理出错: {str(e)}")
            yield pb2.InferenceResponse(
                chunk="", is_finished=True, error=str(e), generated_tokens=0
            )


def serve():
    """启动gRPC服务器"""
    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.maxWorkers))

    # 注册服务
    pb2_grpc.add_InferencerServiceServicer_to_server(
        InferencerServiceServicer(), server
    )

    # 监听端口
    server_address = f"[::]:{config.grpcPort}"
    server.add_insecure_port(server_address)

    # 启动服务器
    logger.info(f"gRPC服务器启动在 {server_address}")
    server.start()

    # 保持服务器运行
    def signal_handler(sig, frame):
        logger.info("收到关闭信号，正在停止gRPC服务器...")
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
