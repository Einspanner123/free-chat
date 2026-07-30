"""
Context Compression Experiment

Usage:
    python run.py                                          # default config
    python run.py --config config.yaml                     # custom config
    python run.py --model longchat --dataset longbench     # override data params
    python run.py --budgets 1024 2048 4096                 # override budgets

Output:
    results/results.json    — structured experiment data
    results/summary.txt     — human-readable summary
    plots/*.png             — visualizations (if matplotlib available)
"""

import argparse
import json
import os
import random
import sys
from typing import List, Dict

# Add project root for config loading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metrics import compute_all_metrics


# =============================================================================
# Synthetic conversation generator
# =============================================================================

_TOPICS = [
    "python programming", "machine learning", "deployment architecture",
    "database design", "API design", "testing strategies", "security best practices",
]


def generate_conversation(num_turns: int = 50, turns_per_topic: int = 7,
                          avg_tokens: int = 250, seed: int = 42) -> List[Dict]:
    """Generate synthetic multi-topic conversation for reproducible benchmarking."""
    random.seed(seed)
    conv = []
    topic_idx = 0

    from metrics import ENTITIES

    for turn in range(num_turns):
        topic = _TOPICS[topic_idx % len(_TOPICS)]
        entities = ENTITIES.get(topic, ["concept"])
        entity = random.choice(entities)

        if turn % 2 == 0:
            role = "user"
            n_words = random.randint(int(avg_tokens * 0.6), int(avg_tokens * 1.2))
            content = f"Tell me about {entity} in {topic}. " + (
                f"This is a detailed question about {entity} and its role in the broader context of {topic}. "
                * max(1, n_words // 15)
            )
        else:
            role = "assistant"
            n_words = random.randint(int(avg_tokens * 0.8), int(avg_tokens * 1.4))
            content = f"Let me explain {entity}. " + (
                f"In the context of {topic}, {entity} refers to an important concept that you should understand. "
                * max(1, n_words // 15)
            )

        conv.append({
            "turn": turn, "role": role, "topic": topic, "content": content,
            "tokens": len(content.split()) + int(len(content) * 0.25),
        })

        if (turn + 1) % turns_per_topic == 0:
            topic_idx += 1

    return conv


# =============================================================================
# Compression methods
# =============================================================================

def full_context(conversation: List[Dict]) -> List[Dict]:
    """Baseline: keep everything verbatim."""
    return [{**t, "compressed_tokens": t["tokens"]} for t in conversation]


def truncation(conversation: List[Dict], budget: int) -> List[Dict]:
    """Keep most recent turns within budget, discard everything older."""
    total = 0
    kept = []
    for turn in reversed(conversation):
        tok = turn["tokens"]
        if total + tok <= budget:
            kept.insert(0, {**turn, "compressed_tokens": tok})
            total += tok
        else:
            break
    return kept


def hierarchical_compression(conversation: List[Dict], budget: int,
                             levels: List[Dict] = None) -> List[Dict]:
    """Tiered compression by recency."""
    if levels is None:
        levels = [
            {"name": "verbatim", "max_turns": 5, "max_chars": None},
            {"name": "light", "max_turns": 20, "max_chars": 100},
            {"name": "medium", "max_turns": 50, "max_chars": 50},
            {"name": "heavy", "max_turns": float('inf'), "max_chars": 0},
        ]

    total = 0
    compressed = []

    for i, turn in enumerate(conversation):
        turn_num = len(conversation) - i

        for level in levels:
            if turn_num <= level["max_turns"]:
                if level["max_chars"] is None:
                    new_content = turn["content"]
                    new_tokens = turn["tokens"]
                elif level["max_chars"] > 0:
                    new_content = turn["content"][:level["max_chars"]]
                    new_tokens = len(new_content.split()) + 1
                else:
                    new_content = "[compressed from earlier topic]"
                    new_tokens = 4
                break
        else:
            new_content = "[compressed]"
            new_tokens = 2

        if total + new_tokens <= budget:
            compressed.append({**turn, "content": new_content, "compressed_tokens": new_tokens})
            total += new_tokens
        else:
            break

    return compressed


# =============================================================================
# Experiment runner
# =============================================================================

def run_experiment(
    num_turns: int = 50,
    turns_per_topic: int = 7,
    avg_tokens: int = 250,
    budgets: List[int] = None,
    seed: int = 42,
) -> Dict:
    """Run full experiment across compression methods and budgets."""
    if budgets is None:
        budgets = [1024, 2048, 4096, 8192]

    conversation = generate_conversation(num_turns, turns_per_topic, avg_tokens, seed)
    original_tokens = sum(t["tokens"] for t in conversation)

    results = {
        "experiment": "context_compression",
        "config": {
            "num_turns": num_turns,
            "turns_per_topic": turns_per_topic,
            "avg_tokens": avg_tokens,
            "budgets": budgets,
            "seed": seed,
        },
        "conversation_stats": {
            "total_turns": len(conversation),
            "total_tokens": original_tokens,
        },
        "methods": {},
    }

    for budget in budgets:
        # Full context (baseline)
        full = full_context(conversation)
        full_metrics = compute_all_metrics(conversation, full)
        results["methods"].setdefault("full_context", []).append({
            "budget": budget, **full_metrics
        })

        # Truncation
        trunc = truncation(conversation, budget)
        trunc_metrics = compute_all_metrics(conversation, trunc)
        results["methods"].setdefault("truncation", []).append({
            "budget": budget, **trunc_metrics
        })

        # Hierarchical compression
        comp = hierarchical_compression(conversation, budget)
        comp_metrics = compute_all_metrics(conversation, comp)
        results["methods"].setdefault("hierarchical", []).append({
            "budget": budget, **comp_metrics
        })

    return results


def format_summary(results: Dict) -> str:
    """Generate human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("Context Compression Experiment Results")
    lines.append("=" * 70)
    lines.append("")

    cs = results["conversation_stats"]
    lines.append(f"Conversation: {cs['total_turns']} turns, {cs['total_tokens']} tokens")
    lines.append(f"Budgets: {results['config']['budgets']}")
    lines.append("")

    # Find best compression for each budget
    for budget in results["config"]["budgets"]:
        lines.append(f"\n--- Budget: {budget} tokens ---")
        lines.append(f"{'Method':<25} {'Tokens':<10} {'Ratio':<10} {'Recall':<10} {'Turns':<8} {'Latency':<10}")
        lines.append("-" * 73)
        for method_key, method_label in [("full_context", "Full Context"), ("truncation", "Truncation"), ("hierarchical", "Hierarchical")]:
            entries = results["methods"].get(method_key, [])
            entry = next((e for e in entries if e["budget"] == budget), None)
            if entry:
                lines.append(
                    f"{method_label:<25} {entry['compressed_tokens']:<10} "
                    f"{entry['compression_ratio']:<10.1%} {entry['entity_recall']:<10.1%} "
                    f"{entry['turns_kept']:<8} {entry['compressed_latency_ms']:<10.0f}ms"
                )

    return "\n".join(lines)


def save_results(results: Dict, out_dir: str = "results"):
    """Save results as JSON and summary text."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(format_summary(results))
    print(f"Results saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Context Compression Experiment")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--num-turns", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    # Load config if specified
    budgets = args.budgets

    print(f"Context Compression Experiment")
    print(f"  Turns: {args.num_turns}, Seed: {args.seed}")
    if budgets:
        print(f"  Budgets: {budgets}")
    print()

    results = run_experiment(
        num_turns=args.num_turns,
        budgets=budgets,
        seed=args.seed,
    )

    save_results(results, args.out)
    print()
    print(format_summary(results))
    print()


if __name__ == "__main__":
    main()
