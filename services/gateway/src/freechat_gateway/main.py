import os

import uvicorn

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


def build_app():  # type: ignore[no-untyped-def]
    secret = os.environ.get("FREECHAT_CACHE_SALT_SECRET", "s" * 32).encode()
    scheduler_target = os.environ.get("FREECHAT_SCHEDULER_TARGET")
    scheduler = GrpcSchedulerClient(scheduler_target) if scheduler_target else None
    return create_app(
        GatewayConfig(
            api_keys=_keys_from_environment(),
            cache_salt_secret=secret,
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
