from benchmarks.serving_baseline import Endpoint, prometheus_counter, select_endpoint


def test_prometheus_counter_sums_matching_series_only() -> None:
    payload = """
# HELP vllm:prefix_cache_hits_total cached tokens
vllm:prefix_cache_hits_total{engine="0",model_name="a"} 16
vllm:prefix_cache_hits_total{engine="1",model_name="a"} 32
vllm:prefix_cache_hits_created{engine="0"} 100
"""
    assert prometheus_counter(payload, "vllm:prefix_cache_hits_total") == 48


def test_round_robin_is_deterministic() -> None:
    endpoints = [Endpoint("a", "http://a", "0"), Endpoint("b", "http://b", "1")]
    assert [select_endpoint("round-robin", endpoints, turn).name for turn in range(4)] == [
        "a",
        "b",
        "a",
        "b",
    ]


def test_least_load_uses_name_as_stable_tie_breaker() -> None:
    endpoints = [
        Endpoint("b", "http://b", "1", active_requests=0),
        Endpoint("a", "http://a", "0", active_requests=0),
    ]
    assert select_endpoint("least-load", endpoints, 0).name == "a"
    endpoints[1].active_requests = 2
    assert select_endpoint("least-load", endpoints, 0).name == "b"
