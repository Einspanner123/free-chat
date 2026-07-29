"""
Data processing module for fine-tuning.

Supports multiple data formats (ShareGPT, Alpaca, ChatML),
template conversion, data splitting, and statistics.
"""

import json
import os
import random
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter


# Maps ShareGPT "from" field to standard roles
_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


class DataProcessor:
    """Process training data in various formats."""

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_sharegpt(self, data: List[dict]) -> List[dict]:
        """Load ShareGPT-format data.

        Expected format:
            [{"conversations": [{"from": "human", "value": "..."}, ...]}]
        """
        if not data:
            return []
        results = []
        for entry in data:
            if "conversations" not in entry:
                raise ValueError(
                    "ShareGPT entry missing 'conversations' field"
                )
            conversations = entry["conversations"]
            if not isinstance(conversations, list):
                raise ValueError("'conversations' must be a list")

            messages = []
            # Add system prompt if present
            if "system" in entry and entry["system"]:
                messages.append({
                    "role": "system",
                    "content": str(entry["system"]),
                })

            for turn in conversations:
                if "value" not in turn:
                    raise ValueError(
                        f"ShareGPT conversation turn missing 'value' field: {turn}"
                    )
                from_role = turn.get("from", "")
                if from_role not in _SHAREGPT_ROLE_MAP:
                    raise ValueError(
                        f"Unsupported 'from' role '{from_role}'. "
                        f"Supported: {list(_SHAREGPT_ROLE_MAP.keys())}"
                    )
                role = _SHAREGPT_ROLE_MAP[from_role]
                content = str(turn["value"]).strip()
                if not content:
                    continue
                messages.append({"role": role, "content": content})

            # Validate: should end with assistant
            if messages and messages[-1]["role"] != "assistant":
                # Truncate trailing user messages
                while messages and messages[-1]["role"] != "assistant":
                    messages.pop()

            if not messages:
                continue

            # Check all non-empty
            if all(m.get("content") for m in messages):
                results.append({"messages": messages})

        return results

    def load_alpaca(self, data: List[dict]) -> List[dict]:
        """Load Alpaca-format data.

        Expected format:
            [{"instruction": "...", "input": "...", "output": "..."}]
        """
        if not data:
            return []
        results = []
        for entry in data:
            instruction = entry.get("instruction", "").strip()
            input_text = entry.get("input", "").strip()
            output = entry.get("output", "").strip()
            if not output:
                continue
            # Combine instruction and input
            user_content = instruction
            if input_text:
                user_content += "\n\n" + input_text
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            results.append({"messages": messages})
        return results

    def load_chatml(self, data: List[dict]) -> List[dict]:
        """Load ChatML-format data.

        Expected format:
            [{"messages": [{"role": "...", "content": "..."}, ...]}]
        """
        if not data:
            return []
        results = []
        for entry in data:
            if "messages" not in entry:
                continue
            messages = entry["messages"]
            if not messages:
                continue
            # Validate role/content
            valid = True
            for m in messages:
                if "role" not in m or "content" not in m:
                    valid = False
                    break
            if valid:
                results.append({"messages": messages})
        return results

    def load_file(
        self,
        path: str,
        format: str = "sharegpt",
    ) -> List[dict]:
        """Load data from a JSON or JSONL file.

        Args:
            path: Path to the data file.
            format: One of "sharegpt", "alpaca", "chatml".

        Returns:
            List of processed entries, each with a "messages" key.
        """
        valid_formats = {"sharegpt", "alpaca", "chatml"}
        if format not in valid_formats:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {sorted(valid_formats)}"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)

        if format == "sharegpt":
            return self.load_sharegpt(data)
        elif format == "alpaca":
            return self.load_alpaca(data)
        elif format == "chatml":
            return self.load_chatml(data)
        else:
            # Should not reach here due to early validation
            raise ValueError(f"Unexpected format: {format}")

    # -----------------------------------------------------------------------
    # Template conversion
    # -----------------------------------------------------------------------

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenizer=None,
    ) -> str:
        """Apply chat template to messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            tokenizer: Optional tokenizer with apply_chat_template.

        Returns:
            Formatted string.
        """
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback template
        parts = []
        for m in messages:
            role = m["role"]
            content = str(m.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def format_for_training(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """Format messages for training.

        Returns a dict with "input" (full context) and optionally "output"
        or "labels" for loss computation.
        """
        if not messages:
            return {}
        # Input = everything except the last assistant message
        # Output = the last assistant message
        input_msgs = messages[:-1]
        output_msg = messages[-1] if messages else {}

        input_text = self.apply_template(input_msgs) if input_msgs else ""
        output_text = str(output_msg.get("content", ""))

        return {
            "input": input_text,
            "output": output_text,
        }

    # -----------------------------------------------------------------------
    # Data splitting
    # -----------------------------------------------------------------------

    def split(
        self,
        data: List[dict],
        eval_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[dict], List[dict]]:
        """Split data into training and evaluation sets.

        Args:
            data: List of data entries.
            eval_ratio: Proportion for evaluation (0.0 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            (train_data, eval_data)
        """
        if not data:
            return [], []
        rng = random.Random(seed)
        indices = list(range(len(data)))
        rng.shuffle(indices)
        eval_count = max(1, int(len(data) * eval_ratio)) if eval_ratio > 0 else 0
        eval_indices = set(indices[:eval_count])
        train = [data[i] for i in indices if i not in eval_indices]
        eval_data = [data[i] for i in indices if i in eval_indices]
        return train, eval_data

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample(
        self,
        data: List[dict],
        n: int,
        seed: int = 42,
    ) -> List[dict]:
        """Sample n entries from data.

        Args:
            data: List of data entries.
            n: Number of samples.
            seed: Random seed for reproducibility.

        Returns:
            Sampled entries.
        """
        if not data or n <= 0:
            return []
        n = min(n, len(data))
        rng = random.Random(seed)
        indices = list(range(len(data)))
        rng.shuffle(indices)
        return [data[i] for i in indices[:n]]

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def compute_statistics(
        self,
        data: List[dict],
        tokenizer=None,
    ) -> Dict[str, Any]:
        """Compute dataset statistics.

        Args:
            data: List of entries with "messages" key.
            tokenizer: Optional tokenizer for token counting.

        Returns:
            Dict with statistics.
        """
        if not data:
            return {"num_examples": 0}

        total_messages = 0
        total_input_tokens = 0
        total_output_tokens = 0
        role_counter = Counter()

        for entry in data:
            messages = entry.get("messages", [])
            total_messages += len(messages)
            for m in messages:
                role = m.get("role", "unknown")
                role_counter[role] += 1
                content = str(m.get("content", ""))
                if tokenizer is not None and hasattr(tokenizer, "encode"):
                    tokens = len(tokenizer.encode(content))
                else:
                    tokens = len(content) // 2
                if role == "assistant":
                    total_output_tokens += tokens
                else:
                    total_input_tokens += tokens

        num_examples = len(data)
        avg_messages = total_messages / num_examples if num_examples > 0 else 0

        return {
            "num_examples": num_examples,
            "total_turns": total_messages,
            "avg_messages_per_example": round(avg_messages, 2),
            "avg_input_tokens": total_input_tokens // num_examples if num_examples > 0 else 0,
            "avg_output_tokens": total_output_tokens // num_examples if num_examples > 0 else 0,
            "role_distribution": dict(role_counter),
        }
