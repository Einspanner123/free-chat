"""
Metrics for context compression evaluation.

Computes:
- Compression ratio
- Entity recall (information preservation)
- Latency estimation
- Turn preservation rate
"""

from typing import List, Dict, Set


# Entity dictionary used by the synthetic conversation generator
ENTITIES = {
    "python programming": ["Python", "Django", "Flask", "async/await", "type hints",
                          "list comprehension", "generator", "decorator"],
    "machine learning": ["PyTorch", "Transformer", "attention", "embedding",
                        "loss function", "gradient descent", "overfitting"],
    "deployment architecture": ["Kubernetes", "Docker", "load balancer", "auto-scaling",
                               "health check", "blue-green", "canary"],
    "database design": ["PostgreSQL", "index", "sharding", "replication",
                       "ACID", "normalization", "query optimization"],
    "API design": ["REST", "gRPC", "OpenAPI", "versioning", "pagination",
                  "rate limiting", "idempotency"],
    "testing strategies": ["unit test", "integration test", "mock", "fixture",
                          "coverage", "property-based", "snapshot"],
    "security best practices": ["OAuth", "JWT", "encryption", "XSS", "CSRF",
                               "SQL injection", "CORS"],
}


def extract_entities(conversation: List[Dict]) -> Set[str]:
    """Extract all entity mentions from a conversation."""
    entities = set()
    for turn in conversation:
        content = turn.get("content", "")
        if content.startswith("[compressed"):
            continue
        topic = turn.get("topic", "")
        for entity in ENTITIES.get(topic, []):
            if entity.lower() in content.lower():
                entities.add(entity)
    return entities


def compute_entity_recall(original: List[Dict], compressed: List[Dict]) -> float:
    """Fraction of original entities preserved after compression."""
    orig_entities = extract_entities(original)
    if not orig_entities:
        return 1.0
    comp_entities = extract_entities(compressed)
    return len(orig_entities & comp_entities) / len(orig_entities)


def compute_compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """Compression ratio: 0 = no reduction, 1 = 100% reduction."""
    if original_tokens == 0:
        return 0.0
    return 1.0 - (compressed_tokens / original_tokens)


def estimate_latency_ms(total_tokens: int, ms_per_token: float = 0.5) -> float:
    """Estimate prompt encoding latency based on token count."""
    return total_tokens * ms_per_token


def compute_all_metrics(original: List[Dict], compressed: List[Dict]) -> Dict:
    """Compute all metrics for a compression result."""
    orig_tokens = sum(t.get("tokens", 0) for t in original)
    comp_tokens = sum(t.get("compressed_tokens", t.get("tokens", 0)) for t in compressed)

    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "compression_ratio": round(compute_compression_ratio(orig_tokens, comp_tokens), 4),
        "entity_recall": round(compute_entity_recall(original, compressed), 4),
        "turns_original": len(original),
        "turns_kept": len(compressed),
        "turn_preservation": round(len(compressed) / max(len(original), 1), 4),
        "original_latency_ms": round(estimate_latency_ms(orig_tokens), 1),
        "compressed_latency_ms": round(estimate_latency_ms(comp_tokens), 1),
        "latency_reduction": round(1.0 - comp_tokens / max(orig_tokens, 1), 4),
    }
