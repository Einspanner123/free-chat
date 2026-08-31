from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthContext:
    tenant_id: str
    subject: str
    method: str


class APIKeyAuthenticator:
    """Small development authenticator with production-safe comparison.

    OIDC validation is attached behind the same interface. Raw keys are held
    only in process memory for the Compose fixture and never written to etcd.
    """

    def __init__(self, keys: dict[str, str]) -> None:
        self._hashes = {
            hashlib.sha256(key.encode()).digest(): tenant for tenant, key in keys.items()
        }

    def authenticate(self, authorization: str | None, x_api_key: str | None) -> AuthContext:
        candidate = x_api_key
        if candidate is None and authorization is not None:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                candidate = value
        if not candidate:
            raise PermissionError("missing API credential")
        digest = hashlib.sha256(candidate.encode()).digest()
        for expected, tenant in self._hashes.items():
            if hmac.compare_digest(digest, expected):
                return AuthContext(tenant_id=tenant, subject=f"api-key:{tenant}", method="api-key")
        raise PermissionError("invalid API credential")
