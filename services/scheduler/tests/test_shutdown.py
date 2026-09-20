from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
async def test_scheduler_signal_exits_cleanly(signum: signal.Signals) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("FREECHAT_") and key not in {"ETCD_ENDPOINT", "NATS_URL"}
    }
    env.update(
        FREECHAT_WORKER_TOKEN="t" * 32,
        FREECHAT_SCHEDULER_LISTEN="127.0.0.1:0",
        PYTHONPATH=os.pathsep.join(sys.path),
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "from freechat_scheduler.grpc_server import run; run()",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    try:
        async with asyncio.timeout(15):
            while True:
                line = await process.stdout.readline()
                assert line, "Scheduler exited before readiness"
                if b"scheduler_ready" in line:
                    break
        process.send_signal(signum)
        output, _ = await asyncio.wait_for(process.communicate(), timeout=15)
        assert process.returncode == 0, output.decode()
        assert b"scheduler_stop_requested" in output
        assert b"scheduler_shutdown_complete" in output
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
