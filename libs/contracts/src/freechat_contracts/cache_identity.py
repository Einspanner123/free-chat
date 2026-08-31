from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable


def derive_cache_salt(secret: bytes, tenant_id: str, sharing_scope: str) -> str:
    if len(secret) < 32:
        raise ValueError("cache-salt secret must be at least 32 bytes")
    if not tenant_id or not sharing_scope:
        raise ValueError("tenant_id and sharing_scope are required")
    payload = f"{tenant_id}\x00{sharing_scope}".encode()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def make_cache_identity(
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_hash: str,
    adapter_id: str | None,
    kv_layout: str,
    cache_salt: str,
    token_ids: Iterable[int],
) -> str:
    fields = (
        model_id,
        model_revision,
        tokenizer_revision,
        chat_template_hash,
        adapter_id or "",
        kv_layout,
        cache_salt,
        ",".join(str(token_id) for token_id in token_ids),
    )
    canonical = "\x1f".join(fields).encode()
    return hashlib.sha256(canonical).hexdigest()
