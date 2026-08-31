from typing import Any

from freechat_trace_replay import ParquetTraceArchive


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}

    def put(
        self,
        bucket: str,
        key: str,
        payload: bytes,
        metadata: dict[str, str],
    ) -> None:
        self.objects[(bucket, key)] = payload, metadata


def test_parquet_trace_is_hashed_and_archived() -> None:
    store = MemoryObjectStore()
    archive = ParquetTraceArchive(store, "freechat-evidence")
    records: list[dict[str, Any]] = [
        {"trace_id": "trace", "span": "route", "duration_ms": 3.5},
        {"trace_id": "trace", "span": "decode", "duration_ms": 7.0},
    ]
    manifest = archive.write("trace", records)
    payload, metadata = store.objects[(manifest.bucket, manifest.key)]
    assert payload[:4] == b"PAR1"
    assert manifest.records == 2
    assert metadata["sha256"] == manifest.sha256
