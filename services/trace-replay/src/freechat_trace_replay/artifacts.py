from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


class ObjectStore(Protocol):
    def put(
        self,
        bucket: str,
        key: str,
        payload: bytes,
        metadata: dict[str, str],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    bucket: str
    key: str
    sha256: str
    bytes: int
    records: int


class Boto3ObjectStore:
    def __init__(self, client: Any) -> None:
        self._client = client

    def put(
        self,
        bucket: str,
        key: str,
        payload: bytes,
        metadata: dict[str, str],
    ) -> None:
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            Metadata=metadata,
            ContentType="application/vnd.apache.parquet",
        )


class ParquetTraceArchive:
    def __init__(self, object_store: ObjectStore, bucket: str) -> None:
        self._object_store = object_store
        self._bucket = bucket

    def write(self, trace_id: str, records: list[dict[str, Any]]) -> ArtifactManifest:
        if not records:
            raise ValueError("a trace artifact must contain at least one record")
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        output = io.BytesIO()
        table = pa.Table.from_pylist(records)
        pq.write_table(table, output, compression="zstd", version="2.6")
        payload = output.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        day = datetime.now(UTC).strftime("%Y/%m/%d")
        key = f"traces/{day}/{trace_id}.parquet"
        self._object_store.put(
            self._bucket,
            key,
            payload,
            {"sha256": digest, "records": str(len(records)), "trace-id": trace_id},
        )
        return ArtifactManifest(
            bucket=self._bucket,
            key=key,
            sha256=digest,
            bytes=len(payload),
            records=len(records),
        )
