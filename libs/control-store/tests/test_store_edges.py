from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from freechat_control_store import CompareFailed, EtcdHttpStore, InMemoryStore
from freechat_control_store.store import _prefix_end


@pytest.mark.parametrize(
    "prefix,end", [(b"/a", b"/b"), (b"a\xff", b"b"), (b"\xff", b"\0"), (b"", b"\0")]
)
def test_prefix_scan_upper_bound(prefix: bytes, end: bytes) -> None:
    assert _prefix_end(prefix) == end


async def test_missing_in_memory_delete_does_not_create_entry() -> None:
    store = InMemoryStore()
    assert await store.compare_and_delete("unknown", 0) is False
    assert await store.list_prefix("") == ()


@pytest.mark.parametrize("empty", [False, True])
async def test_etcd_range_decodes_revision_and_prefix(empty: bool) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert base64.b64decode(payload["key"]) == b"/key"
        if "range_end" in payload:
            assert base64.b64decode(payload["range_end"]) == b"/kez"
        return httpx.Response(
            200,
            json={
                "kvs": []
                if empty
                else [{"key": "L2tleQ==", "value": "dmFsdWU=", "mod_revision": "42"}]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        store = EtcdHttpStore("http://etcd/", client=client)
        item = await store.get("/key")
        found = await store.list_prefix("/key")
        assert (item is None) == empty
        assert len(found) == (0 if empty else 1)
        if item:
            assert (item.key, item.value, item.revision) == ("/key", b"value", 42)
        await store.close()
        assert not client.is_closed


async def test_owned_etcd_client_is_closed() -> None:
    store = EtcdHttpStore("http://never-contact")
    await store.close()
    assert store._client.is_closed


@pytest.mark.parametrize(
    "payload", [{}, {"header": {}}, {"header": {"revision": 0}}, {"header": {"revision": -1}}]
)
async def test_write_without_valid_revision_is_not_a_receipt(payload: dict[str, Any]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises((ValueError, KeyError)):
            await EtcdHttpStore("http://etcd", client=client).put("k", b"v")


async def test_delete_conflict_is_not_success() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"succeeded": False})
        )
    ) as client:
        with pytest.raises(CompareFailed):
            await EtcdHttpStore("http://etcd", client=client).compare_and_delete("k", 2)
