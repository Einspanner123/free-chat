import os

import uvicorn
from fastapi import FastAPI

from freechat_gateway.app import GatewayConfig, create_app
from freechat_gateway.routing import GrpcSchedulerClient


def _keys_from_environment() -> dict[str, str]:
    raw = os.environ.get("FREECHAT_API_KEYS", "development:development-key")
    pairs: dict[str, str] = {}
    for item in raw.split(","):
        tenant, separator, key = item.partition(":")
        if not separator or not tenant or not key:
            raise ValueError("FREECHAT_API_KEYS must contain tenant:key pairs")
        pairs[tenant] = key
    return pairs


def build_app() -> FastAPI:
    secret = os.environ.get("FREECHAT_CACHE_SALT_SECRET", "s" * 32).encode()
    scheduler_target = os.environ.get("FREECHAT_SCHEDULER_TARGET", "").strip()
    if not scheduler_target:
        raise ValueError("FREECHAT_SCHEDULER_TARGET is required; static routing is test-only")
    scheduler = GrpcSchedulerClient(scheduler_target)
    return create_app(
        GatewayConfig(
            api_keys=_keys_from_environment(),
            cache_salt_secret=secret,
            origin_node_id=os.environ.get("FREECHAT_ORIGIN_NODE_ID") or None,
            default_worker_endpoint=os.environ.get(
                "FREECHAT_DEFAULT_WORKER_ENDPOINT", "http://worker:8000"
            ),
        ),
        scheduler=scheduler,
    )


def run() -> None:
    uvicorn.run(
        "freechat_gateway.main:build_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
