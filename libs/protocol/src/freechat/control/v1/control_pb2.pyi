import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RequestContext(_message.Message):
    __slots__ = ("request_id", "idempotency_key", "tenant_id", "traceparent", "deadline", "schema_version")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    TRACEPARENT_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    idempotency_key: str
    tenant_id: str
    traceparent: str
    deadline: _timestamp_pb2.Timestamp
    schema_version: int
    def __init__(self, request_id: _Optional[str] = ..., idempotency_key: _Optional[str] = ..., tenant_id: _Optional[str] = ..., traceparent: _Optional[str] = ..., deadline: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., schema_version: _Optional[int] = ...) -> None: ...

class AgentHints(_message.Message):
    __slots__ = ("task_id", "session_id", "agent_id", "parent_agent_id", "branch_id", "parent_branch_id", "turn_id", "call_id", "lifecycle", "prefix_scope", "reuse_class", "expected_reuse_probability", "expected_resume_ms", "ttl_ms", "priority", "deadline_ms", "expected_output_tokens", "privacy_domain", "allow_preemption", "allow_kv_offload", "allow_remote_worker", "confidence", "source", "harness_id", "harness_version")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    BRANCH_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_BRANCH_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    CALL_ID_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_FIELD_NUMBER: _ClassVar[int]
    PREFIX_SCOPE_FIELD_NUMBER: _ClassVar[int]
    REUSE_CLASS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_REUSE_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_RESUME_MS_FIELD_NUMBER: _ClassVar[int]
    TTL_MS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PRIVACY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    ALLOW_PREEMPTION_FIELD_NUMBER: _ClassVar[int]
    ALLOW_KV_OFFLOAD_FIELD_NUMBER: _ClassVar[int]
    ALLOW_REMOTE_WORKER_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    HARNESS_ID_FIELD_NUMBER: _ClassVar[int]
    HARNESS_VERSION_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    session_id: str
    agent_id: str
    parent_agent_id: str
    branch_id: str
    parent_branch_id: str
    turn_id: str
    call_id: str
    lifecycle: str
    prefix_scope: str
    reuse_class: str
    expected_reuse_probability: float
    expected_resume_ms: int
    ttl_ms: int
    priority: int
    deadline_ms: int
    expected_output_tokens: int
    privacy_domain: str
    allow_preemption: bool
    allow_kv_offload: bool
    allow_remote_worker: bool
    confidence: float
    source: str
    harness_id: str
    harness_version: str
    def __init__(self, task_id: _Optional[str] = ..., session_id: _Optional[str] = ..., agent_id: _Optional[str] = ..., parent_agent_id: _Optional[str] = ..., branch_id: _Optional[str] = ..., parent_branch_id: _Optional[str] = ..., turn_id: _Optional[str] = ..., call_id: _Optional[str] = ..., lifecycle: _Optional[str] = ..., prefix_scope: _Optional[str] = ..., reuse_class: _Optional[str] = ..., expected_reuse_probability: _Optional[float] = ..., expected_resume_ms: _Optional[int] = ..., ttl_ms: _Optional[int] = ..., priority: _Optional[int] = ..., deadline_ms: _Optional[int] = ..., expected_output_tokens: _Optional[int] = ..., privacy_domain: _Optional[str] = ..., allow_preemption: _Optional[bool] = ..., allow_kv_offload: _Optional[bool] = ..., allow_remote_worker: _Optional[bool] = ..., confidence: _Optional[float] = ..., source: _Optional[str] = ..., harness_id: _Optional[str] = ..., harness_version: _Optional[str] = ...) -> None: ...

class RouteRequest(_message.Message):
    __slots__ = ("context", "hints", "model", "input_tokens", "output_tokens", "cache_key", "streaming")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    HINTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHE_KEY_FIELD_NUMBER: _ClassVar[int]
    STREAMING_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    hints: AgentHints
    model: str
    input_tokens: int
    output_tokens: int
    cache_key: str
    streaming: bool
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., hints: _Optional[_Union[AgentHints, _Mapping]] = ..., model: _Optional[str] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., cache_key: _Optional[str] = ..., streaming: _Optional[bool] = ...) -> None: ...

class CostBreakdown(_message.Message):
    __slots__ = ("queue_ms", "prefill_ms", "decode_ms", "cache_ms", "network_ms", "cold_start_ms", "deadline_risk", "eviction_externality", "total", "affinity_credit_ms")
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_MS_FIELD_NUMBER: _ClassVar[int]
    DECODE_MS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_MS_FIELD_NUMBER: _ClassVar[int]
    COLD_START_MS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_RISK_FIELD_NUMBER: _ClassVar[int]
    EVICTION_EXTERNALITY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    AFFINITY_CREDIT_MS_FIELD_NUMBER: _ClassVar[int]
    queue_ms: float
    prefill_ms: float
    decode_ms: float
    cache_ms: float
    network_ms: float
    cold_start_ms: float
    deadline_risk: float
    eviction_externality: float
    total: float
    affinity_credit_ms: float
    def __init__(self, queue_ms: _Optional[float] = ..., prefill_ms: _Optional[float] = ..., decode_ms: _Optional[float] = ..., cache_ms: _Optional[float] = ..., network_ms: _Optional[float] = ..., cold_start_ms: _Optional[float] = ..., deadline_risk: _Optional[float] = ..., eviction_externality: _Optional[float] = ..., total: _Optional[float] = ..., affinity_credit_ms: _Optional[float] = ...) -> None: ...

class RouteDecision(_message.Message):
    __slots__ = ("decision_id", "worker_id", "endpoint", "worker_generation", "cost", "rejected_candidates", "lease_ttl_ms", "topology_generation", "strategy")
    DECISION_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    WORKER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    REJECTED_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    LEASE_TTL_MS_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_GENERATION_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    decision_id: str
    worker_id: str
    endpoint: str
    worker_generation: int
    cost: CostBreakdown
    rejected_candidates: _containers.RepeatedScalarFieldContainer[str]
    lease_ttl_ms: int
    topology_generation: int
    strategy: str
    def __init__(self, decision_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., endpoint: _Optional[str] = ..., worker_generation: _Optional[int] = ..., cost: _Optional[_Union[CostBreakdown, _Mapping]] = ..., rejected_candidates: _Optional[_Iterable[str]] = ..., lease_ttl_ms: _Optional[int] = ..., topology_generation: _Optional[int] = ..., strategy: _Optional[str] = ...) -> None: ...

class LeaseRequest(_message.Message):
    __slots__ = ("context", "decision_id", "worker_id", "worker_generation")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    DECISION_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    decision_id: str
    worker_id: str
    worker_generation: int
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., decision_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_generation: _Optional[int] = ...) -> None: ...

class WorkerRegistration(_message.Message):
    __slots__ = ("context", "worker_id", "generation", "endpoint", "capabilities_json")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_JSON_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    worker_id: str
    generation: int
    endpoint: str
    capabilities_json: str
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., worker_id: _Optional[str] = ..., generation: _Optional[int] = ..., endpoint: _Optional[str] = ..., capabilities_json: _Optional[str] = ...) -> None: ...

class WorkerHeartbeat(_message.Message):
    __slots__ = ("context", "worker_id", "generation", "telemetry_json")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_JSON_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    worker_id: str
    generation: int
    telemetry_json: str
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., worker_id: _Optional[str] = ..., generation: _Optional[int] = ..., telemetry_json: _Optional[str] = ...) -> None: ...

class CacheEventBatch(_message.Message):
    __slots__ = ("context", "worker_id", "worker_generation", "cache_generation", "events_json")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    CACHE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    EVENTS_JSON_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    worker_id: str
    worker_generation: int
    cache_generation: int
    events_json: bytes
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., worker_id: _Optional[str] = ..., worker_generation: _Optional[int] = ..., cache_generation: _Optional[int] = ..., events_json: _Optional[bytes] = ...) -> None: ...

class CacheActionRequest(_message.Message):
    __slots__ = ("context", "worker_id", "worker_generation", "cache_generation", "actions_json")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    CACHE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_JSON_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    worker_id: str
    worker_generation: int
    cache_generation: int
    actions_json: bytes
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., worker_id: _Optional[str] = ..., worker_generation: _Optional[int] = ..., cache_generation: _Optional[int] = ..., actions_json: _Optional[bytes] = ...) -> None: ...

class ReplayRequest(_message.Message):
    __slots__ = ("context", "manifest_uri", "policy", "mode")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_URI_FIELD_NUMBER: _ClassVar[int]
    POLICY_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    manifest_uri: str
    policy: str
    mode: str
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., manifest_uri: _Optional[str] = ..., policy: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...

class Operation(_message.Message):
    __slots__ = ("operation_id", "status", "detail")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    status: str
    detail: str
    def __init__(self, operation_id: _Optional[str] = ..., status: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...
