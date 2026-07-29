"""
KV Cache Manager: block-based allocation pool, eviction policies, prefix cache.

Design:
  - BlockPool: fixed-size block pool with allocate/free, tracks per-request ownership
  - EvictionPolicies: LRU, SlidingWindow, AttentionWeighted — pluggable
  - KVCacheManager: combines pool + eviction + prefix cache into one interface
  - PrefixCache: hash-keyed prefix reuse with reference counting
"""

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Dict, Set, Tuple


# =============================================================================
# Block Pool
# =============================================================================

class BlockPool:
    """Fixed-size block pool for KV cache allocation.

    Blocks are identified by integer IDs. Each block stores KV states for
    a fixed number of tokens (block_size_tokens). The pool tracks which
    request owns each block, enabling proper free and reuse.
    """

    def __init__(self, total_blocks: int = 64, block_size_tokens: int = 256):
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self._free_blocks: Set[int] = set(range(total_blocks))
        self._allocated: Dict[str, List[int]] = {}  # request_id → [block_ids]

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self._free_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def fragmentation_pct(self) -> float:
        """Estimate fragmentation as 1 - (largest free run / total free)."""
        if not self._free_blocks:
            return 0.0
        sorted_free = sorted(self._free_blocks)
        longest_run = 1
        current_run = 1
        for i in range(1, len(sorted_free)):
            if sorted_free[i] == sorted_free[i-1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        if len(sorted_free) == 0:
            return 0.0
        return 1.0 - (longest_run / len(sorted_free))

    def available_blocks(self) -> int:
        return self.free_blocks

    def allocate(self, request_id: str, num_blocks: int = 1) -> List[int]:
        """Allocate blocks for a request.

        Args:
            request_id: Unique request identifier.
            num_blocks: Number of blocks requested.

        Returns:
            List of allocated block IDs (may be fewer than num_blocks if pool is exhausted).
        """
        if request_id not in self._allocated:
            self._allocated[request_id] = []

        to_allocate = min(num_blocks, len(self._free_blocks))
        if to_allocate <= 0:
            return []

        acquired = []
        for _ in range(to_allocate):
            block_id = self._free_blocks.pop()
            acquired.append(block_id)

        self._allocated[request_id].extend(acquired)
        return acquired

    def free(self, request_id: str):
        """Free all blocks owned by a request."""
        if request_id not in self._allocated:
            return
        for block_id in self._allocated[request_id]:
            self._free_blocks.add(block_id)
        del self._allocated[request_id]

    def owner_of(self, block_id: int) -> Optional[str]:
        """Return the request ID that owns a block, or None."""
        for req_id, blocks in self._allocated.items():
            if block_id in blocks:
                return req_id
        return None


# =============================================================================
# Eviction Policies
# =============================================================================

class EvictionPolicy(ABC):
    @abstractmethod
    def record_access(self, request_id: str):
        """Record that a request was accessed."""
        ...

    @abstractmethod
    def record_position(self, token_id: str, position: int):
        """Record a token's position (for sliding window)."""
        ...

    @abstractmethod
    def evict(self, num_blocks: int) -> List[str]:
        """Return request IDs to evict."""
        ...

    @abstractmethod
    def remove(self, request_id: str):
        """Remove a request from tracking (when it finishes)."""
        ...


class LRUEvictionPolicy(EvictionPolicy):
    """Evict requests that were accessed least recently."""

    def __init__(self):
        self._access_order: OrderedDict = OrderedDict()

    def record_access(self, request_id: str):
        self._access_order[request_id] = True
        self._access_order.move_to_end(request_id)

    def record_position(self, token_id: str, position: int):
        pass  # LRU doesn't use positional info

    def evict(self, num_blocks: int) -> List[str]:
        victims = []
        for _ in range(num_blocks):
            if not self._access_order:
                break
            req_id, _ = self._access_order.popitem(last=False)
            victims.append(req_id)
        return victims

    def remove(self, request_id: str):
        self._access_order.pop(request_id, None)


class SlidingWindowEvictionPolicy(EvictionPolicy):
    """Keep only the last W tokens in cache; older tokens are eligible for eviction."""

    def __init__(self, window_size: int = 4096):
        self.window_size = window_size
        self._positions: Dict[str, int] = {}

    def record_access(self, request_id: str):
        pass  # Sliding window doesn't use access frequency

    def record_position(self, token_id: str, position: int):
        self._positions[token_id] = position

    def evict(self, num_blocks: int) -> List[str]:
        """Return tokens whose position is outside the window."""
        if not self._positions:
            return []
        max_pos = max(self._positions.values())
        min_window_start = max(0, max_pos - self.window_size)
        victims = []
        for token_id, pos in list(self._positions.items()):
            if pos < min_window_start:
                victims.append(token_id)
                del self._positions[token_id]
            if len(victims) >= num_blocks:
                break
        return victims

    def remove(self, request_id: str):
        self._positions.pop(request_id, None)


class AttentionWeightedEvictionPolicy(EvictionPolicy):
    """Track which tokens receive the most attention; evict lowest-attention tokens first.

    Reference: H2O (Heavy-Hitter Oracle, Zhang et al., NeurIPS 2023).
    The insight is that a small fraction of tokens account for most attention scores.
    """

    def __init__(self, keep_ratio: float = 0.3):
        self.keep_ratio = keep_ratio
        self._scores: Dict[str, float] = {}

    def record_access(self, request_id: str):
        pass

    def record_position(self, token_id: str, position: int):
        pass

    def update_score(self, token_id: str, attention_score: float):
        """Update the cumulative attention score for a token."""
        self._scores[token_id] = self._scores.get(token_id, 0.0) + attention_score

    def evict(self, num_blocks: int) -> List[str]:
        if not self._scores:
            return []
        # Sort by attention score ascending, return lowest-scoring
        sorted_tokens = sorted(self._scores.items(), key=lambda x: x[1])
        victims = [t for t, _ in sorted_tokens[:num_blocks]]
        for t in victims:
            del self._scores[t]
        return victims

    def remove(self, request_id: str):
        self._scores.pop(request_id, None)


# =============================================================================
# Prefix Cache
# =============================================================================

class PrefixCache:
    """Cache for shared prompt prefixes with LRU eviction."""

    def __init__(self, max_prefixes: int = 64):
        self.max_prefixes = max_prefixes
        self._entries: OrderedDict = OrderedDict()  # hash → {prefix, blocks, ref_count}

    @property
    def size(self) -> int:
        return len(self._entries)

    def store(self, prefix: str, block_ids: List[int]):
        """Store prefix→blocks mapping. Replaces existing entry for same prefix."""
        key = self._hash(prefix)
        self._entries[key] = {
            "prefix": prefix,
            "blocks": list(block_ids),
        }
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_prefixes:
            self._entries.popitem(last=False)

    def lookup(self, prefix: str) -> Optional[List[int]]:
        """Exact match lookup. Returns block IDs or None."""
        key = self._hash(prefix)
        entry = self._entries.get(key)
        if entry and entry["prefix"] == prefix:
            self._entries.move_to_end(key)
            return entry["blocks"]
        return None

    def match_prefix(self, text: str) -> Tuple[Optional[List[int]], float]:
        """Find the best partial prefix match.

        Returns:
            (block_ids, score) where score is fraction of characters matched.
        """
        best = None
        best_score = 0.0
        for entry in self._entries.values():
            score = self._prefix_score(entry["prefix"], text)
            if score > best_score:
                best_score = score
                best = entry["blocks"]
        if best and best_score > 0.5:
            return best, best_score
        return None, 0.0

    def clear(self):
        self._entries.clear()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _prefix_score(prefix: str, text: str) -> float:
        if not prefix or not text:
            return 0.0
        common = 0
        for a, b in zip(prefix, text):
            if a == b:
                common += 1
            else:
                break
        if common == 0:
            return 0.0
        return common / len(prefix)


# =============================================================================
# KV Cache Manager (unified interface)
# =============================================================================

_EVICTION_POLICIES = {
    "lru": LRUEvictionPolicy,
    "sliding_window": SlidingWindowEvictionPolicy,
    "attention_weighted": AttentionWeightedEvictionPolicy,
}


class KVCacheManager:
    """Unified manager for KV cache allocation, eviction, and prefix reuse.

    Combines:
      - BlockPool: fixed-size block allocation
      - EvictionPolicy: configurable eviction strategy
      - PrefixCache: prompt prefix reuse

    Usage:
      mgr = KVCacheManager(total_blocks=64)
      mgr.allocate("req_1", num_blocks=4)
      mgr.free("req_1")
    """

    def __init__(
        self,
        total_blocks: int = 64,
        block_size: int = 256,
        eviction_policy: str = "lru",
        max_prefixes: int = 64,
    ):
        self.total_blocks = total_blocks
        self.block_size = block_size

        self._pool = BlockPool(total_blocks=total_blocks, block_size_tokens=block_size)
        self._eviction = self._build_policy(eviction_policy)
        self._prefix_cache = PrefixCache(max_prefixes=max_prefixes)
        self._request_blocks: Dict[str, List[int]] = {}  # request_id → block_ids

    @property
    def free_blocks(self) -> int:
        return self._pool.free_blocks

    @property
    def used_blocks(self) -> int:
        return self._pool.used_blocks

    @property
    def fragmentation_pct(self) -> float:
        return self._pool.fragmentation_pct

    @property
    def prefix_cache_size(self) -> int:
        return self._prefix_cache.size

    @property
    def eviction_policy_name(self) -> str:
        return type(self._eviction).__name__.replace("EvictionPolicy", "").lower()

    def set_eviction_policy(self, policy_name: str, **kwargs):
        """Switch eviction policy at runtime."""
        self._eviction = self._build_policy(policy_name, **kwargs)

    def allocate(self, request_id: str, num_blocks: int = 1) -> List[int]:
        """Allocate blocks for a request. Triggers eviction if pool is full.

        Args:
            request_id: Unique request identifier.
            num_blocks: Number of blocks needed.

        Returns:
            List of allocated block IDs.
        """
        if request_id not in self._request_blocks:
            self._request_blocks[request_id] = []

        # Try direct allocation first
        blocks = self._pool.allocate(request_id, num_blocks)
        allocated = len(blocks)

        # If not enough, trigger eviction
        if allocated < num_blocks:
            needed = num_blocks - allocated
            victims = self._eviction.evict(needed)
            for victim_id in victims:
                self._pool.free(victim_id)
                self._request_blocks.pop(victim_id, None)
                freed = self._pool.allocate(request_id, 1)
                blocks.extend(freed)
                allocated += len(freed)
                if allocated >= num_blocks:
                    break

        self._request_blocks[request_id] = self._request_blocks.get(request_id, []) + blocks
        self._eviction.record_access(request_id)
        return blocks

    def free(self, request_id: str):
        """Free all blocks for a request."""
        self._pool.free(request_id)
        self._request_blocks.pop(request_id, None)
        self._eviction.remove(request_id)

    def store_prefix(self, prefix: str, block_ids: List[int]):
        """Cache KV blocks for a prompt prefix."""
        self._prefix_cache.store(prefix, block_ids)

    def lookup_prefix(self, prefix: str) -> Optional[List[int]]:
        """Look up cached blocks for an exact prefix match."""
        return self._prefix_cache.lookup(prefix)

    def match_prefix(self, text: str) -> Tuple[Optional[List[int]], float]:
        """Find the best partial prefix match."""
        return self._prefix_cache.match_prefix(text)

    def stats(self) -> dict:
        return {
            "total_blocks": self._pool.total_blocks,
            "used_blocks": self._pool.used_blocks,
            "free_blocks": self._pool.free_blocks,
            "fragmentation_pct": round(self._pool.fragmentation_pct * 100, 1),
            "prefix_cache_entries": self._prefix_cache.size,
            "active_requests": len(self._request_blocks),
            "eviction_policy": self.eviction_policy_name,
        }

    @staticmethod
    def _build_policy(name: str, **kwargs) -> EvictionPolicy:
        cls = _EVICTION_POLICIES.get(name)
        if cls is None:
            raise ValueError(f"Unknown eviction policy: {name}. Available: {list(_EVICTION_POLICIES.keys())}")
        return cls(**kwargs)
