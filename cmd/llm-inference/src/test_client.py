import statistics
import time
import uuid

import grpc
import llm_inference_pb2 as pb2
import llm_inference_pb2_grpc as pb2_grpc

"""
==================================================

测试消息 1: 请介绍一下人工智能的发展历程。
  接收响应数: 514
  总耗时: 17060.35 ms
  首响应延迟: 1001.72 ms

测试消息 2: 解释一下量子计算的基本原理。
  接收响应数: 514
  总耗时: 17746.81 ms
  首响应延迟: 42.40 ms

测试消息 3: Python和Go语言各有什么优缺点？
  接收响应数: 514
  总耗时: 14749.72 ms
  首响应延迟: 52.49 ms

测试消息 4: 描述一下宇宙大爆炸理论的主要内容。
  接收响应数: 514
  总耗时: 17014.70 ms
  首响应延迟: 42.83 ms

测试消息 5: 我是一名智力低下的硕士生, 请你解释DDIM的扩散原理, 每一步每一个符号都必须合理解释, 并且用中文回答
  接收响应数: 514
  总耗时: 14844.37 ms
  首响应延迟: 59.70 ms

==================================================
测试完成，统计结果:
  测试次数: 5
  平均延迟: 239.83 ms
  最小延迟: 42.40 ms
  最大延迟: 1001.72 ms
  延迟标准差: 425.97 ms
"""


def run_test():
    # 连接到gRPC服务器
    channel = grpc.insecure_channel("localhost:50051")
    stub = pb2_grpc.InferencerServiceStub(channel)

    # 测试参数
    test_messages = [
        "请介绍一下人工智能的发展历程。",
        "解释一下量子计算的基本原理。",
        "Python和Go语言各有什么优缺点？",
        "描述一下宇宙大爆炸理论的主要内容。",
        "我是一名智力低下的硕士生, 请你解释DDIM的扩散原理, 每一步每一个符号都必须合理解释, 并且用中文回答",
    ]

    # 存储统计数据
    latencies = []

    print("开始测试gRPC流式推理服务...")
    print("=" * 50)

    for i, message in enumerate(test_messages):
        print(f"\n测试消息 {i + 1}: {message}")

        # 创建会话ID
        session_id = str(uuid.uuid4())

        # 记录整体请求开始时间
        start_time = time.time()

        # 创建请求生成器
        def request_generator():
            yield pb2.InferenceRequest(
                session_id=session_id, message=message, temperature=0.7
            )

        # 发送流式请求并接收响应
        response_count = 0
        first_response_time = None

        try:
            responses = stub.StreamInference(request_generator())

            for response in responses:
                # 记录第一个响应的时间
                if first_response_time is None:
                    first_response_time = time.time()

                response_count += 1
                if response.chunk:
                    print(response.chunk, end="", flush=True)
                if response.is_finished:
                    break

        except grpc.RpcError as e:
            print(f"RPC错误: {e.code()}, details: {e.details()}")
            continue
        except Exception as e:
            print(f"其他错误: {e}")
            continue
        print("\n")

        # 计算延迟
        end_time = time.time()
        total_duration = (end_time - start_time) * 1000  # 转换为毫秒
        first_response_latency = (
            (first_response_time - start_time) * 1000 if first_response_time else None
        )

        # 输出本次测试结果
        print(f"  接收响应数: {response_count}")
        print(f"  总耗时: {total_duration:.2f} ms")
        if first_response_latency:
            print(f"  首响应延迟: {first_response_latency:.2f} ms")
            latencies.append(first_response_latency)

        # 控制测试频率，避免过于频繁
        time.sleep(1)

    # 输出统计结果
    print("\n" + "=" * 50)
    print("测试完成，统计结果:")

    if latencies:
        print(f"  测试次数: {len(latencies)}")
        print(f"  平均延迟: {statistics.mean(latencies):.2f} ms")
        print(f"  最小延迟: {min(latencies):.2f} ms")
        print(f"  最大延迟: {max(latencies):.2f} ms")
        print(
            f"  延迟标准差: {statistics.stdev(latencies):.2f} ms"
            if len(latencies) > 1
            else "  延迟标准差: N/A"
        )
    else:
        print("  没有成功收集到延迟数据")


if __name__ == "__main__":
    run_test()
