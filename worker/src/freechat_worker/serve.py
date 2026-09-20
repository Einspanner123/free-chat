"""Managed native vLLM HTTP server with durable admission and execution RPC."""

from __future__ import annotations

import asyncio
import fcntl
import importlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import grpc
import uvicorn
from freechat.control.execution import LocalRequestExecutionService
from freechat.control.v1 import control_pb2_grpc
from pydantic import SecretStr

from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.native_serving import (
    AdmissionMiddleware,
    NativeEngineClient,
    NativeExecutionBackend,
)

LOGGER = logging.getLogger(__name__)


async def serve(args: Any) -> None:
    token = os.environ.get("FREECHAT_WORKER_TOKEN", "")
    if len(token) < 32:
        raise ValueError("FREECHAT_WORKER_TOKEN must contain at least 32 characters")
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", args.worker_id) is None:
        raise ValueError("worker ID must be a safe filesystem identifier")
    directory = Path(args.state_dir) / args.worker_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One managed incarnation per persistent worker directory, before allocating GPU.
    with (directory / "runtime.owner").open("a+b") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = importlib.import_module("vllm.entrypoints.openai.api_server")
        async with api.build_async_engine_client(args) as engine:
            backend = NativeExecutionBackend(engine)
            proxy = NativeEngineClient(engine, backend)
            instance = str(uuid4())
            driver = DurableExecutionDriver(
                directory / f"{args.worker_generation}.sqlite",
                backend,
                worker_id=args.worker_id,
                generation=args.worker_generation,
                engine_instance_id=instance,
                create=True,
            )
            control = grpc.aio.server()
            control_pb2_grpc.add_RequestExecutionServiceServicer_to_server(  # type: ignore[no-untyped-call]
                LocalRequestExecutionService(
                    driver,
                    worker_id=args.worker_id,
                    generation=args.worker_generation,
                    engine_instance_id=instance,
                    token=SecretStr(token),
                ),
                control,
            )
            if not control.add_insecure_port(f"127.0.0.1:{args.control_port}"):
                driver.close()
                raise RuntimeError("worker control port unavailable")
            try:
                tasks = await proxy.get_supported_tasks()
                app = api.build_app(args, tasks, engine.model_config)
                await api.init_app_state(proxy, app.state, args, tasks)
                wrapped = AdmissionMiddleware(app, driver=driver, backend=backend, token=token)
                await control.start()
                LOGGER.info(
                    "worker ready worker=%s generation=%s engine=%s http=%s:%s control=%s",
                    args.worker_id,
                    args.worker_generation,
                    instance,
                    args.host,
                    args.port,
                    args.control_port,
                )
                server = uvicorn.Server(
                    uvicorn.Config(
                        wrapped,
                        host=args.host,
                        port=args.port,
                        log_level=args.uvicorn_log_level,
                    )
                )
                await server.serve()
            finally:
                await control.stop(grace=5)
                driver.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    api = importlib.import_module("vllm.entrypoints.openai.api_server")
    parser = api.make_arg_parser(api.FlexibleArgumentParser(description=__doc__))
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-generation", type=int, default=time.time_ns())
    parser.add_argument("--state-dir", default="/var/lib/freechat")
    parser.add_argument("--control-port", type=int, default=50052)
    parser.set_defaults(async_scheduling=False, host="127.0.0.1")
    args = parser.parse_args()
    if args.worker_generation < 1:
        parser.error("worker generation must be positive")
    api.validate_parsed_serve_args(args)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
