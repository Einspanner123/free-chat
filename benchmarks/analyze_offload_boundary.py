"""Paired, stratified analysis. Never upgrades this boundary probe to acceptance."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    index = int(position)
    fraction = position - index
    return ordered[index] * (1 - fraction) + ordered[min(index + 1, len(ordered) - 1)] * fraction


def paired_p95(baseline: list[float], candidate: list[float]) -> dict[str, Any]:
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("requires matched pairs")
    rng = random.Random(42)
    differences = []
    for _ in range(10000):
        indices = rng.choices(range(len(baseline)), k=len(baseline))
        differences.append(
            percentile([candidate[i] for i in indices], 0.95)
            - percentile([baseline[i] for i in indices], 0.95)
        )
    return {
        "baseline_p95": percentile(baseline, 0.95),
        "preoffload_p95": percentile(candidate, 0.95),
        "difference_candidate_minus_baseline": percentile(candidate, 0.95)
        - percentile(baseline, 0.95),
        "paired_bootstrap_95_interval": [
            percentile(differences, 0.025),
            percentile(differences, 0.975),
        ],
        "pairs": len(baseline),
        "draws": 10000,
        "seed": 42,
    }


def analyze(root: Path) -> dict[str, Any]:
    index = json.loads((root / "index.json").read_text())
    groups: dict[str, dict[str, dict[bool, dict[str, Any]]]] = {}
    hashes = []
    for item in index:
        path = root / item["name"] / "result.json"
        record = json.loads(path.read_text())
        for artifact in record["artifacts"]:
            if (
                hashlib.sha256((path.parent / artifact["path"]).read_bytes()).hexdigest()
                != artifact["sha256"]
            ):
                raise ValueError("raw artifact hash mismatch")
        hashes.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if item["warmup"]:
            continue
        key = record["harness"] + "/" + record["scenario"]
        pair = item["name"].rsplit("-", 2)[-2]
        arms = groups.setdefault(key, {}).setdefault(pair, {})
        arm = bool(record["preoffload"])
        if arm in arms:
            raise ValueError("duplicate arm")
        arms[arm] = record
    output = {}
    for key, pairs in groups.items():
        if any(set(pair) != {False, True} for pair in pairs.values()):
            raise ValueError("missing pair arm")
        baseline = [pair[False] for pair in pairs.values()]
        candidate = [pair[True] for pair in pairs.values()]

        def values(records: list[dict[str, Any]], metric: str) -> list[float]:
            if metric in {"adjusted_task_ms", "raw_task_ms"}:
                return [float(item[metric]) for item in records]
            return [float(item["calls"][1]["deltas"][metric]) for item in records]

        metrics = {
            metric: paired_p95(values(baseline, metric), values(candidate, metric))
            for metric in (
                "adjusted_task_ms",
                "raw_task_ms",
                "vllm:time_to_first_token_seconds_sum",
                "vllm:request_prefill_kv_computed_tokens_sum",
            )
        }
        stored = sum(
            call["deltas"]["vllm:kv_offload_store_size_sum"]
            for record in candidate
            for call in record["calls"]
        )
        loaded = sum(
            call["deltas"]["vllm:kv_offload_load_size_sum"]
            for record in candidate
            for call in record["calls"]
        )
        output[key] = {
            "metrics": metrics,
            "stored_bytes": stored,
            "loaded_bytes": loaded,
            "terminal_unrestored_bytes": stored - loaded,
            "strict_correct": {
                "baseline": sum(item["correct"] for item in baseline),
                "preoffload": sum(item["correct"] for item in candidate),
            },
            "semantic_heading_present": {
                "baseline": sum(item["semantic_heading_present"] for item in baseline),
                "preoffload": sum(item["semantic_heading_present"] for item in candidate),
            },
        }
    return {
        "groups": output,
        "source_hashes": hashes,
        "final_ab_acceptance": False,
        "resume_performance_claim_admissible": False,
        "limitations": [
            "Two Harnesses, one forced coding task, one GPU and one small model.",
            "A predictor and B implementation are not evaluated.",
            "Wrong exact-format answers remain in all measured samples.",
            "20 pairs provide limited P95 precision; no multiple-test correction.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output = args.root / "analysis.json"
    if output.exists():
        raise ValueError("analysis exists; preserve prior result")
    report = analyze(args.root)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
