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


@pytest.mark.parametrize("cas", [False, True])
async def test_write_returns_own_value_and_commit_revision_without_later_read(cas: bool) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v3/kv/range":
            # Another writer already replaced our committed value.
            return httpx.Response(
                200,
                json={"kvs": [{"key": "aw==", "value": "b3RoZXI=", "mod_revision": "12"}]},
            )
        return httpx.Response(200, json={"header": {"revision": "11"}, "succeeded": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        store = EtcdHttpStore("http://etcd", client=client)
        item = (
            await store.compare_and_put("k", 10, b"ours") if cas else await store.put("k", b"ours")
        )
        assert item.value == b"ours" and item.revision == 11
        assert calls == ["/v3/kv/txn" if cas else "/v3/kv/put"]


@pytest.mark.parametrize("deleted", [0, 1])
async def test_delete_reports_whether_matching_key_existed(deleted: int) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "succeeded": True,
                "responses": [{"response_delete_range": {"deleted": str(deleted)}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        store = EtcdHttpStore("http://etcd", client=client)
        assert await store.compare_and_delete("k", 0) is bool(deleted)
