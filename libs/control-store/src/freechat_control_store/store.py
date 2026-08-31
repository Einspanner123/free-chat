from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass(frozen=True, slots=True)
class KeyValue:
    key: str
    value: bytes
    revision: int


class CompareFailed(RuntimeError):
    pass


class KeyValueStore(Protocol):
    async def get(self, key: str) -> KeyValue | None: ...

    async def put(self, key: str, value: bytes) -> KeyValue: ...

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue: ...


class InMemoryStore:
    """Deterministic test store with etcd-style monotonic revisions."""

    def __init__(self) -> None:
        self._revision = 0
        self._values: dict[str, KeyValue] = {}

    async def get(self, key: str) -> KeyValue | None:
        return self._values.get(key)

    async def put(self, key: str, value: bytes) -> KeyValue:
        self._revision += 1
        item = KeyValue(key=key, value=value, revision=self._revision)
        self._values[key] = item
        return item

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        current = self._values.get(key)
        current_revision = 0 if current is None else current.revision
        if current_revision != expected_revision:
            raise CompareFailed(
                f"revision mismatch for {key}: expected {expected_revision}, got {current_revision}"
            )
        return await self.put(key, value)


class EtcdHttpStore:
    """Minimal etcd API client for authoritative generation and decision state.

    It uses the official JSON gRPC gateway. Lifecycle ownership remains with
    the caller; no hidden retry is performed around compare-and-swap writes.
    """

    def __init__(self, endpoint: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=5.0)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get(self, key: str) -> KeyValue | None:
        response = await self._client.post(
            f"{self._endpoint}/v3/kv/range",
            json={"key": _encode(key)},
        )
        response.raise_for_status()
        entries = response.json().get("kvs", [])
        if not entries:
            return None
        entry = entries[0]
        return KeyValue(
            key=_decode(entry["key"]).decode(),
            value=_decode(entry["value"]),
            revision=int(entry["mod_revision"]),
        )

    async def put(self, key: str, value: bytes) -> KeyValue:
        response = await self._client.post(
            f"{self._endpoint}/v3/kv/put",
            json={"key": _encode(key), "value": _encode(value)},
        )
        response.raise_for_status()
        item = await self.get(key)
        if item is None:
            raise RuntimeError(f"etcd acknowledged put but key is absent: {key}")
        return item

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        compare_target = "CREATE" if expected_revision == 0 else "MOD"
        response = await self._client.post(
            f"{self._endpoint}/v3/kv/txn",
            json={
                "compare": [
                    {
                        "key": _encode(key),
                        "target": compare_target,
                        "result": "EQUAL",
                        "create_revision" if expected_revision == 0 else "mod_revision": str(
                            expected_revision
                        ),
                    }
                ],
                "success": [
                    {"request_put": {"key": _encode(key), "value": _encode(value)}}
                ],
                "failure": [],
            },
        )
        response.raise_for_status()
        if not response.json().get("succeeded", False):
            raise CompareFailed(f"revision mismatch for {key}")
        item = await self.get(key)
        if item is None:
            raise RuntimeError(f"etcd transaction succeeded but key is absent: {key}")
        return item


def _encode(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.b64encode(raw).decode()


def _decode(value: str) -> bytes:
    return base64.b64decode(value, validate=True)
