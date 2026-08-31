import asyncio

import httpx
import pytest
from freechat_control_store import CompareFailed, EtcdHttpStore, InMemoryStore


def test_in_memory_compare_and_put_rejects_stale_revision() -> None:
    async def scenario() -> None:
        store = InMemoryStore()
        created = await store.compare_and_put("/workers/ross", 0, b"one")
        updated = await store.compare_and_put("/workers/ross", created.revision, b"two")
        assert updated.revision > created.revision
        with pytest.raises(CompareFailed):
            await store.compare_and_put("/workers/ross", created.revision, b"stale")

    asyncio.run(scenario())


def test_etcd_transaction_failure_is_explicit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v3/kv/txn"
        return httpx.Response(200, json={"succeeded": False})

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        store = EtcdHttpStore("http://etcd", client=client)
        with pytest.raises(CompareFailed):
            await store.compare_and_put("/leases/request", 7, b"decision")
        await client.aclose()

    asyncio.run(scenario())


def test_prefix_listing_and_fenced_delete() -> None:
    async def scenario() -> None:
        store = InMemoryStore()
        first = await store.put("/workers/a", b"a")
        await store.put("/workers/b", b"b")
        await store.put("/leases/c", b"c")
        assert [item.key for item in await store.list_prefix("/workers/")] == [
            "/workers/a",
            "/workers/b",
        ]
        with pytest.raises(CompareFailed):
            await store.compare_and_delete(first.key, first.revision + 1)
        assert await store.compare_and_delete(first.key, first.revision)
        assert await store.get(first.key) is None

    asyncio.run(scenario())
