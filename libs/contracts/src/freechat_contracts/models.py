from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Lifecycle(StrEnum):
    SPAWN = "spawn"
    ACTIVE = "active"
    TOOL_WAIT = "tool_wait"
    RESUME = "resume"
    TERMINAL = "terminal"
    CANCELLED = "cancelled"


class PrefixScope(StrEnum):
    GLOBAL = "global"
    HARNESS = "harness"
    TASK = "task"
    AGENT = "agent"
    BRANCH = "branch"
    PRIVATE = "private"


class ReuseClass(StrEnum):
    IMMUTABLE_SHARED = "immutable_shared"
    GROWING_HISTORY = "growing_history"
    EPHEMERAL_REASONING = "ephemeral_reasoning"
    UNKNOWN = "unknown"


class HintSource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class CacheEventKind(StrEnum):
    ALLOCATE = "allocate"
    FREE = "free"
    HIT = "hit"
    EVICT = "evict"
    OFFLOAD = "offload"
    LOAD = "load"


class CacheActionKind(StrEnum):
    RETAIN = "retain"
    OFFLOAD_CPU = "offload_cpu"
    OFFLOAD_NVME = "offload_nvme"
    PREFETCH = "prefetch"
    EVICT = "evict"


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class AgentHints(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: int = Field(default=1, ge=1)
    harness_id: str = Field(min_length=1, max_length=128)
    harness_version: str | None = Field(default=None, max_length=64)
    task_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, max_length=256)
    agent_id: str = Field(min_length=1, max_length=256)
    parent_agent_id: str | None = Field(default=None, max_length=256)
    branch_id: str = Field(default="main", min_length=1, max_length=256)
    parent_branch_id: str | None = Field(default=None, max_length=256)
    turn_id: str | None = Field(default=None, max_length=256)
    call_id: str = Field(default_factory=lambda: str(uuid4()), max_length=256)
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    prefix_scope: PrefixScope = PrefixScope.PRIVATE
    reuse_class: ReuseClass = ReuseClass.UNKNOWN
    expected_reuse_probability: Probability = 0.0
    expected_resume_ms: int | None = Field(default=None, ge=0, le=86_400_000)
    ttl_ms: int = Field(default=300_000, ge=1_000, le=86_400_000)
    priority: int = Field(default=0, ge=-20, le=20)
    deadline_ms: int | None = Field(default=None, ge=1)
    expected_output_tokens: int | None = Field(default=None, ge=1)
    privacy_domain: str = Field(default="default", min_length=1, max_length=128)
    allow_preemption: bool = True
    allow_kv_offload: bool = True
    allow_remote_worker: bool = True
    source: HintSource = HintSource.EXPLICIT
    confidence: Confidence = 1.0
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_semantics(self) -> AgentHints:
        if self.source is HintSource.INFERRED and self.confidence >= 1.0:
            raise ValueError("inferred hints must have confidence below 1.0")
        if (
            self.lifecycle in {Lifecycle.TOOL_WAIT, Lifecycle.RESUME}
            and self.expected_resume_ms is None
        ):
            raise ValueError("tool_wait and resume require expected_resume_ms")
        if self.parent_agent_id is not None and self.parent_agent_id == self.agent_id:
            raise ValueError("an agent cannot be its own parent")
        if self.parent_branch_id is not None and self.parent_branch_id == self.branch_id:
            raise ValueError("a branch cannot be its own parent")
        if len(self.metadata) > 32:
            raise ValueError("metadata is limited to 32 entries")
        return self


ALLOWED_LIFECYCLE_TRANSITIONS: dict[Lifecycle, frozenset[Lifecycle]] = {
    Lifecycle.SPAWN: frozenset({Lifecycle.ACTIVE, Lifecycle.CANCELLED}),
    Lifecycle.ACTIVE: frozenset(
        {Lifecycle.ACTIVE, Lifecycle.TOOL_WAIT, Lifecycle.TERMINAL, Lifecycle.CANCELLED}
    ),
    Lifecycle.TOOL_WAIT: frozenset({Lifecycle.RESUME, Lifecycle.CANCELLED}),
    Lifecycle.RESUME: frozenset(
        {Lifecycle.ACTIVE, Lifecycle.TOOL_WAIT, Lifecycle.TERMINAL, Lifecycle.CANCELLED}
    ),
    Lifecycle.TERMINAL: frozenset(),
    Lifecycle.CANCELLED: frozenset(),
}


def validate_lifecycle_transition(current: Lifecycle, target: Lifecycle) -> None:
    if target not in ALLOWED_LIFECYCLE_TRANSITIONS[current]:
        raise ValueError(f"invalid lifecycle transition: {current} -> {target}")


class ModelCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    tokenizer_revision: str
    architecture: str
    attention: str
    is_moe: bool = False
    max_context_tokens: int = Field(gt=0)
    dtype: str
    quantization: str | None = None
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    supports_prefix_cache: bool = True
    supports_kv_offload: bool = False
    supports_tool_calling: bool = True


class WorkerCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    generation: int = Field(ge=1)
    endpoint: str
    node_id: str
    gpu_id: str
    gpu_name: str
    compute_capability: str
    total_vram_bytes: int = Field(gt=0)
    p2p_domain: str
    network_domain: str
    models: tuple[ModelCapability, ...]
    allow_remote_requests: bool = True


class WorkerTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    generation: int
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    healthy: bool = True
    draining: bool = False
    queue_depth: int = Field(default=0, ge=0)
    active_requests: int = Field(default=0, ge=0)
    free_vram_bytes: int = Field(ge=0)
    cached_prefixes: frozenset[str] = Field(default_factory=frozenset)
    estimated_prefill_tokens_per_second: float = Field(default=1.0, gt=0)
    estimated_decode_tokens_per_second: float = Field(default=1.0, gt=0)
    network_rtt_ms: float = Field(default=0.0, ge=0)
    network_bandwidth_bytes_per_second: float = Field(default=1.0, gt=0)
    cache_load_bytes_per_second: float = Field(default=1.0, gt=0)


class RequestProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    model_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(gt=0)
    estimated_kv_bytes: int = Field(default=0, ge=0)
    cache_key: str | None = None
    local_node_id: str | None = None
    hints: AgentHints


class CandidateCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    queue_ms: float = Field(ge=0)
    prefill_ms: float = Field(ge=0)
    decode_ms: float = Field(ge=0)
    cache_ms: float = Field(ge=0)
    network_ms: float = Field(ge=0)
    cold_start_ms: float = Field(ge=0)
    deadline_risk: float = Field(ge=0)
    eviction_externality: float = Field(ge=0)
    affinity_credit_ms: float = Field(ge=0)
    total_ms: float


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    worker_id: str
    worker_generation: int
    endpoint: str
    selected: CandidateCost
    candidates: tuple[CandidateCost, ...]
    rejected: dict[str, tuple[str, ...]]
    topology_generation: int
    lease_ttl_ms: int = Field(default=30_000, ge=1_000)


class CacheEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: CacheEventKind
    tenant_id: str
    worker_id: str
    worker_generation: int
    cache_generation: int
    block_id: str
    cache_key: str
    token_count: int = Field(ge=0)
    bytes: int = Field(ge=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: CacheActionKind
    block_id: str
    worker_id: str
    worker_generation: int
    cache_generation: int
    reason: str
    estimated_cost_ms: float = Field(ge=0)
