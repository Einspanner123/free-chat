import json
import os
import signal
import socket
import sys
import urllib
from concurrent import futures
from typing import Iterator

import time

import grpc
import llm_inference_pb2 as pb2
import llm_inference_pb2_grpc as pb2_grpc
from chat_model import ChatModel
from config import config
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from loguru import logger


def get_local_ip():
    env_ip = os.getenv("ADVERTISE_IP")
    if env_ip:
        return env_ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def register_consul(service_id, name, address, port, consul_addr):
    url = f"http://{consul_addr}/v1/agent/service/register"
    payload = {
        "ID": service_id,
        "Name": name,
        "Tags": [name, "api", "v1"],
        "Address": address,
        "Port": port,
        "Check": {
            "GRPC": f"{address}:{port}",
            "GRPCUseTLS": False,
            "Interval": "10s",
            "Timeout": "3s",
            "DeregisterCriticalServiceAfter": "1m",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="PUT", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as _:
            logger.info(f"{name}注册成功")
            return True
    except Exception as e:
        logger.error(f"{name}注册失败: {e}")
        return False


def deregister_consul(service_id, consul_addr):
    url = f"http://{consul_addr}/v1/agent/service/deregister/{service_id}"
    req = urllib.request.Request(url, method="PUT")
    try:
        urllib.request.urlopen(req, timeout=5).close()
        logger.info("Consul注销成功")
        return True
    except Exception as e:
        logger.error(f"Consul注销失败: {e}")
        return False


# 实现 gRPC 服务类
class InferencerServiceServicer(pb2_grpc.InferencerServiceServicer):
    def __init__(self):
        # 这里可以初始化模型和其他资源
        logger.info(f"初始化服务，使用模型: {config.modelName}")
        self.model = ChatModel(
            model_name=config.modelName,
            max_new_tokens=config.maxTokens,
            temperature=config.temperature,
            repeat_penalty=config.repetitionPenalty,
            top_p=config.topP,
            top_k=config.topK,
        )

    def StreamInference(
        self, request_iterator: Iterator[pb2.InferenceRequest], context
    ) -> Iterator[pb2.InferenceResponse]:
        """实现流式推理方法"""
        logger.info("接收到流式推理请求")

        try:
            # 收集请求信息
            session_id = None
            messages: str = ""
            temperature = config.temperature
            for request in request_iterator:
                session_id = request.session_id
                if request.message:
                    messages += str(request.message)
                else:
                    continue
                logger.info(
                    f"接收到消息: session_id={session_id}, message_length={len(request.message)}"
                )
                gen_tokens = 0
                start_time = time.time()
                try:
                    streamer = self.model.GetStreamer(msg=messages)
                    logger.info("start getting response.")
                    for chunk in streamer:
                        gen_tokens += len(self.model.tokenizer.tokenize(chunk))
                        yield pb2.InferenceResponse(
                            chunk=chunk,
                            is_finished=False,
                            error="",
                            generated_tokens=gen_tokens,
                        )
                    
                    duration = time.time() - start_time
                    tps = gen_tokens / duration if duration > 0 else 0
                    logger.info(f"生成完成: tokens={gen_tokens}, time={duration:.2f}s, tps={tps:.2f}")

                    # 发送结束信号
                    yield pb2.InferenceResponse(
                        chunk="",
                        is_finished=True,
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

        except Exception as e:
            logger.error(f"流式处理消息出错: {str(e)}")
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

    # 注册健康服务
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    service_name = os.getenv("SERVER_NAME", config.serverName)
    local_ip = get_local_ip()
    service_id = f"{service_name}-{local_ip}-{config.grpcPort}"
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)

    consul_addr = os.getenv("CONSUL_ADDRESS", "localhost:8500")
    register_consul(service_id, service_name, local_ip, config.grpcPort, consul_addr)
    # 监听端口
    server_address = f"[::]:{config.grpcPort}"
    server.add_insecure_port(server_address)

    # 启动服务器
    logger.info(f"gRPC服务器启动在 {server_address}")
    server.start()

    # 保持服务器运行
    def signal_handler(sig, frame):
        logger.info("收到关闭信号，正在停止gRPC服务器...")
        deregister_consul(service_id, consul_addr)
        
        # 优雅停机，给现有请求 5 秒钟的完成时间
        done_event = server.stop(grace=5)
        done_event.wait(5)
        logger.info("gRPC服务器已停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
