import asyncio

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import ModelCapability, WorkerCapabilities
from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.scheduler import Scheduler


def test_worker_registration_and_route_over_grpc() -> None:
    async def scenario() -> None:
        registry = InMemoryWorkerRegistry()
        server = grpc.aio.server()
        control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
            SchedulerGrpcService(Scheduler(registry), LeaseBook()),
            server,
        )
        control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
            WorkerGrpcService(registry),
            server,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        try:
            worker_stub = control_pb2_grpc.WorkerControlServiceStub(  # type: ignore[no-untyped-call]
                channel
            )
            scheduler_stub = control_pb2_grpc.SchedulerServiceStub(  # type: ignore[no-untyped-call]
                channel
            )
            capabilities = WorkerCapabilities(
                worker_id="ross-a6000",
                generation=1,
                endpoint="http://ross-worker:8000",
                node_id="ross",
                gpu_id="0",
                gpu_name="NVIDIA RTX A6000",
                compute_capability="8.6",
                total_vram_bytes=48 * 1024**3,
                p2p_domain="ross-0",
                network_domain="weak-lan",
                models=(
                    ModelCapability(
                        model_id="Qwen/Qwen2.5-0.5B-Instruct",
                        revision="model-sha",
                        tokenizer_revision="tokenizer-sha",
                        architecture="dense",
                        attention="gqa",
                        max_context_tokens=32_768,
                        dtype="bfloat16",
                    ),
                ),
            )
            registered = await worker_stub.Register(
                control_pb2.WorkerRegistration(
                    context=control_pb2.RequestContext(
                        request_id="register",
                        idempotency_key="register-1",
                        tenant_id="system",
                    ),
                    worker_id=capabilities.worker_id,
                    generation=capabilities.generation,
                    endpoint=capabilities.endpoint,
                    capabilities_json=capabilities.model_dump_json(),
                )
            )
            assert registered.status == "registered"
            route = await scheduler_stub.Route(
                control_pb2.RouteRequest(
                    context=control_pb2.RequestContext(
                        request_id="request",
                        idempotency_key="request-1",
                        tenant_id="tenant-a",
                    ),
                    hints=control_pb2.AgentHints(
                        harness_id="langgraph",
                        task_id="task",
                        agent_id="agent",
                        lifecycle="active",
                        prefix_scope="agent",
                        reuse_class="growing_history",
                        confidence=1,
                        source="explicit",
                        allow_preemption=True,
                        allow_kv_offload=True,
                        allow_remote_worker=True,
                    ),
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    input_tokens=128,
                    output_tokens=32,
                )
            )
            assert route.worker_id == "ross-a6000"
            assert route.worker_generation == 1
            assert route.cost.total > 0
        finally:
            await channel.close()
            await server.stop(grace=None)

    asyncio.run(scenario())
