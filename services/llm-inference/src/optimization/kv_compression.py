"""
MLA-style latent KV compression (inference-side), motivated by kv_lowrank.py.

``research/inference_optimization/run_kv_lowrank.py`` measured that real KV
caches have strong low-rank structure (dimension-dim PCA ≈ MLA's latent
compression; ``implied_latent_save_95`` 0.92→0.44). This module turns that
finding into a *serve-time* optimization: instead of storing the full
``[B, H, S, D]`` K/V per layer, we project each token's K/V to a small latent
(``D -> latent_dim``) via a PCA basis fitted on a calibration corpus and
reconstruct on read. Memory per token drops by ``D / latent_dim`` (e.g. 8× at
latent=16, D=128) at a measurable quality cost (recon error), exactly the
memory × quality tradeoff the benchmark quantifies.

This is the inference-side analog of DeepSeek MLA: the stored KV is a low-dim
latent; attention sees the reconstructed (low-rank) K/V. It is **not** the same
as training-time MLA (the model still computes full-dim attention on the
reconstructed KV), but it compresses the cache the same way.

Gated behind ``kv_compression="mla"`` (default ``none``). Mutually exclusive
with KV eviction. A calibrated basis (``.pt`` from
``run_kv_compression_quality.py``) is recommended; without one a random
orthonormal projection is used (still compresses, quality measured separately).
"""

from typing import List, Optional, Tuple

import torch
from transformers import DynamicCache


def _pca_basis(tensor: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Optimal rank-``latent_dim`` projection basis (right singular vectors).

    ``tensor`` is ``[B, H, S, D]``; flattened to ``[N, D]`` and SVD'd. The top
    ``latent_dim`` right singular vectors form ``U`` (``[D, latent_dim]``) so
    that ``full @ U`` is the best rank-``latent_dim`` approximation (Eckart-Young)
    and ``(full @ U) @ U.T`` reconstructs it.
    """
    D = tensor.shape[-1]
    latent_dim = min(latent_dim, D)
    X = tensor.reshape(-1, D).float()
    # SVD of the data matrix; Vt rows are right singular vectors.
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    del U, S
    basis = Vt[:latent_dim].T.contiguous().to(tensor.dtype)
    return basis


def fit_pca_basis(
    key_cache_list: List[torch.Tensor],
    value_cache_list: List[torch.Tensor],
    latent_dim: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Fit per-layer PCA bases for K and V from calibration caches.

    ``key_cache_list`` / ``value_cache_list`` are per-layer ``[B, H, S, D]``
    tensors (e.g. from a prefilled ``DynamicCache``). Returns ``(basis_k,
    basis_v)``, each a list of ``[D, latent_dim]`` projection matrices.
    """
    if len(key_cache_list) != len(value_cache_list):
        raise ValueError("key/value cache lists must have equal length")
    basis_k: List[torch.Tensor] = []
    basis_v: List[torch.Tensor] = []
    for kc, vc in zip(key_cache_list, value_cache_list):
        basis_k.append(_pca_basis(kc, latent_dim))
        basis_v.append(_pca_basis(vc, latent_dim))
    return basis_k, basis_v


def random_basis(
    n_layers: int,
    head_dim: int,
    latent_dim: int,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Random orthonormal projection bases (fallback when no calibration)."""
    latent_dim = min(latent_dim, head_dim)
    basis_k: List[torch.Tensor] = []
    basis_v: List[torch.Tensor] = []
    for _ in range(n_layers):
        rk = torch.randn(head_dim, latent_dim, generator=generator, device=device)
        rk, _ = torch.linalg.qr(rk)
        rv = torch.randn(head_dim, latent_dim, generator=generator, device=device)
        rv, _ = torch.linalg.qr(rv)
        basis_k.append(rk)
        basis_v.append(rv)
    return basis_k, basis_v


def save_basis(path: str, basis_k: List[torch.Tensor], basis_v: List[torch.Tensor]) -> None:
    torch.save({"basis_k": basis_k, "basis_v": basis_v}, path)


def load_basis(path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj["basis_k"], obj["basis_v"]


class CompressedKVCache(DynamicCache):
    """A ``DynamicCache`` that stores K/V as a latent (MLA-style).

    Each layer keeps ``[B, H, S, latent]`` latent tensors instead of full
    ``[B, H, S, D]``. On ``update``, the stored latent is reconstructed
    (``latent @ U.T``) to recover the past K/V, concatenated with the incoming
    full-dim key/value, and re-projected to latent for storage. Attention sees
    the reconstructed (low-rank) past plus the exact current token — so only
    past tokens pay the compression quality cost.

    Memory is ``latent / D`` per token (e.g. 8× at latent=16, D=128).
    """

    def __init__(
        self,
        basis_k: List[torch.Tensor],
        basis_v: List[torch.Tensor],
        latent_dim: int,
        **kwargs,
    ):
        if len(basis_k) != len(basis_v):
            raise ValueError("basis_k and basis_v must have equal length")
        self._basis_k = basis_k
        self._basis_v = basis_v
        self._latent_dim = latent_dim
        self._lk: List[Optional[torch.Tensor]] = []
        self._lv: List[Optional[torch.Tensor]] = []
        self._logical_seq_len = 0
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Layer latent storage helpers
    # ------------------------------------------------------------------

    def _ensure(self, layer_idx: int):
        while len(self._lk) <= layer_idx:
            self._lk.append(None)
            self._lv.append(None)

    def _stored_len(self, layer_idx: int) -> int:
        if layer_idx >= len(self._lk) or self._lk[layer_idx] is None:
            return 0
        return self._lk[layer_idx].shape[-2]

    # ------------------------------------------------------------------
    # Cache contract overrides
    # ------------------------------------------------------------------

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self._ensure(layer_idx)
        Uk = self._basis_k[layer_idx]
        Uv = self._basis_v[layer_idx]
        if Uk is None:  # no compression configured -> behave as plain cache
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        # reconstruct past from stored latent
        past_k = None
        past_v = None
        if self._lk[layer_idx] is not None:
            past_k = self._lk[layer_idx] @ Uk.T  # [B,H,S_past,D]
            past_v = self._lv[layer_idx] @ Uv.T

        if past_k is None:
            recon_k = key_states
            recon_v = value_states
        else:
            recon_k = torch.cat([past_k, key_states], dim=-2)
            recon_v = torch.cat([past_v, value_states], dim=-2)

        # project back to latent for storage
        self._lk[layer_idx] = recon_k @ Uk  # [B,H,S,latent]
        self._lv[layer_idx] = recon_v @ Uv

        if layer_idx == 0:
            self._logical_seq_len += key_states.shape[-2]
        return recon_k, recon_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._logical_seq_len

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self._logical_seq_len

    def get_mask_sizes(self, query_length: int, layer_idx: int):
        return self._stored_len(layer_idx) + query_length, 0

    def get_max_length(self, layer_idx: int | None = None) -> int:
        # compression does not cap the sequence; return a safe large bound.
        return 1 << 30

    def crop(self, maximum_length: int):
        for li in range(len(self._lk)):
            if self._lk[li] is not None and self._lk[li].shape[-2] > maximum_length:
                self._lk[li] = self._lk[li][..., -maximum_length:, :]
                self._lv[li] = self._lv[li][..., -maximum_length:, :]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def logical_length(self) -> int:
        return self._logical_seq_len

    def latent_length(self, layer_idx: int = 0) -> int:
        return self._stored_len(layer_idx)

    def kv_bytes(self) -> int:
        """Exact bytes held by the compressed (latent) K/V tensors."""
        total = 0
        for lk, lv in zip(self._lk, self._lv):
            if lk is not None:
                total += (lk.numel() + lv.numel()) * lk.element_size()
        return total
