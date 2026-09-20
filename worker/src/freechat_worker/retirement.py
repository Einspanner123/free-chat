"""Seal one stopped managed container's journal, preserving its state and identity.

Run as a short-lived operator container with the SAME state volume mounted at the
same path and access to the local Docker socket. The socket grants daemon access:
a read-only bind mount does not make the Docker API read-only. This implementation
sends only container-inspection GET requests and never stops/removes containers.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import re
import signal
import sqlite3
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import grpc
from freechat.control.execution import LocalRequestExecutionService
from freechat.control.v1 import control_pb2_grpc
from freechat_contracts.execution import ExecutionCommand
from pydantic import SecretStr

from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.runtime import (
    ContainerBinding,
    DockerInspector,
    DockerStopProof,
    stopped_container_proof,
)


def retire_journal(
    state_root: Path,
    *,
    worker_id: str,
    generation: int,
    engine_instance_id: str,
    container_id: str,
    inspector: DockerInspector,
) -> DockerStopProof:
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", worker_id) is None or generation < 1:
        raise ValueError("runtime_worker_identity_invalid")
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise ValueError("runtime_full_container_id_required")
    root = state_root.resolve(strict=True)
    directory = root / worker_id
    path = directory / f"{generation}.sqlite"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("runtime_existing_journal_required")
    with (directory / "runtime.owner").open("a+b") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("runtime_still_owned") from None
        with path.with_suffix(path.suffix + ".owner").open("a+b") as journal_owner:
            try:
                fcntl.flock(journal_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ValueError("execution_journal_already_owned") from None
            db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
            try:
                db.execute("PRAGMA synchronous=FULL")
                row = db.execute("SELECT value FROM meta WHERE id=1").fetchone()
                if row is None or not row[0].startswith("1:"):
                    raise ValueError("execution_journal_uninitialized")
                identity = ExecutionCommand.model_validate_json(row[0][2:])
                if (
                    identity.worker_id,
                    identity.worker_generation,
                    identity.engine_instance_id,
                ) != (worker_id, generation, engine_instance_id):
                    raise ValueError("execution_journal_incarnation_mismatch")
                row = db.execute("SELECT value FROM meta WHERE id=3").fetchone()
                if row is None:
                    raise ValueError("runtime_container_binding_missing")
                binding = ContainerBinding.model_validate_json(row[0])
                inspector.verify_unique_runtime(binding.runtime_id, container_id)
                container = inspector.inspect(container_id)
                observer_id = inspector.container_for_runtime(
                    os.environ.get("FREECHAT_RUNTIME_ID", "")
                )
                observer = inspector.inspect(observer_id)
                proof = stopped_container_proof(
                    container,
                    observer,
                    binding,
                    container_id=container_id,
                    worker_id=worker_id,
                    state_root=str(root),
                    now=datetime.now(UTC),
                )
                existing = db.execute("SELECT value FROM meta WHERE id=2").fetchone()
                if existing is not None:
                    original = DockerStopProof.model_validate_json(existing[0])
                    if original.container_id != container_id:
                        raise ValueError("runtime_retirement_conflict")
                    return original
                with db:
                    db.execute("INSERT INTO meta VALUES (2, ?)", (proof.model_dump_json(),))
                return proof
            finally:
                db.close()


async def serve_retired(
    state_root: Path,
    *,
    worker_id: str,
    generation: int,
    engine_instance_id: str,
    listen: str,
    token: str,
    exclusive_runtime: bool = True,
) -> None:
    """Serve a sealed journal; independent historical RPC may coexist with a new engine."""
    if re.fullmatch(r"127\.0\.0\.1:[1-9][0-9]{0,4}", listen) is None:
        raise ValueError("retired_execution_loopback_required")
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", worker_id) is None or generation < 1:
        raise ValueError("runtime_worker_identity_invalid")
    if len(token) < 32:
        raise ValueError("FREECHAT_WORKER_TOKEN must contain at least 32 characters")
    directory = state_root.resolve(strict=True) / worker_id
    with ExitStack() as stack:
        if exclusive_runtime:
            owner = stack.enter_context((directory / "runtime.owner").open("a+b"))
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ValueError("runtime_still_owned") from None
        driver = DurableExecutionDriver(
            directory / f"{generation}.sqlite",
            None,
            worker_id=worker_id,
            generation=generation,
            engine_instance_id=engine_instance_id,
            retired=True,
        )
        server = grpc.aio.server(options=(("grpc.so_reuseport", 0),))
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        signals: list[signal.Signals] = []
        try:
            control_pb2_grpc.add_RequestExecutionServiceServicer_to_server(  # type: ignore[no-untyped-call]
                LocalRequestExecutionService(
                    driver,
                    worker_id=worker_id,
                    generation=generation,
                    engine_instance_id=engine_instance_id,
                    token=SecretStr(token),
                ),
                server,
            )
            if not server.add_insecure_port(listen):
                raise ValueError("retired_execution_port_unavailable")
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(signum, stop.set)
                signals.append(signum)
            await server.start()
            logging.getLogger(__name__).info(
                "retired_execution_ready worker=%s generation=%s engine=%s",
                worker_id,
                generation,
                engine_instance_id,
            )
            await stop.wait()
        finally:
            for signum in signals:
                loop.remove_signal_handler(signum)
            await server.stop(grace=5)
            driver.close()
            logging.getLogger(__name__).info("retired_execution_stopped")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path("/var/lib/freechat"))
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--engine-instance-id", required=True)
    parser.add_argument("--container-id")
    parser.add_argument("--docker-socket", default="/var/run/docker.sock")
    parser.add_argument(
        "--serve", action="store_true", help="Serve sealed execution receipts, never inference"
    )
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Observe a sealed journal without Docker access; use a separate RPC port",
    )
    parser.add_argument("--listen", default="127.0.0.1:50052")
    args = parser.parse_args()
    if args.observe_only:
        if args.serve or args.container_id:
            parser.error("--observe-only cannot seal a container")
        asyncio.run(
            serve_retired(
                args.state_root,
                worker_id=args.worker_id,
                generation=args.generation,
                engine_instance_id=args.engine_instance_id,
                listen=args.listen,
                token=os.environ.get("FREECHAT_WORKER_TOKEN", ""),
                exclusive_runtime=False,
            )
        )
        return
    if not args.container_id:
        parser.error("--container-id is required to seal a stopped container")
    inspector = DockerInspector(args.docker_socket)
    try:
        proof = retire_journal(
            args.state_root,
            worker_id=args.worker_id,
            generation=args.generation,
            engine_instance_id=args.engine_instance_id,
            container_id=args.container_id,
            inspector=inspector,
        )
        result: dict[str, Any] = {
            "event": "worker_incarnation_retired",
            "worker_id": args.worker_id,
            "generation": args.generation,
            "engine_instance_id": args.engine_instance_id,
            "proof": proof.model_dump(mode="json"),
            "scheduler_capacity_reclaimed": False,
        }
        print(json.dumps(result), flush=True)
    finally:
        inspector.close()
    if args.serve:
        asyncio.run(
            serve_retired(
                args.state_root,
                worker_id=args.worker_id,
                generation=args.generation,
                engine_instance_id=args.engine_instance_id,
                listen=args.listen,
                token=os.environ.get("FREECHAT_WORKER_TOKEN", ""),
            )
        )


if __name__ == "__main__":
    main()
