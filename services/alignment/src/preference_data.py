"""
Preference data loading and construction for DPO training.

Supports the standard chosen/rejected pair format with
validation, multi-turn conversations, and system prompts.
"""

import json
import os
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter


class PreferenceDataLoader:
    """Load and validate preference data for DPO training."""

    def load_standard(self, data: List[Dict]) -> List[Dict]:
        """Load standard preference pair format.

        Expected format:
            [{
                "chosen": [{"role": ..., "content": ...}, ...],
                "rejected": [{"role": ..., "content": ...}, ...],
                "system": "..." (optional)
            }]
        """
        if not data:
            return []
        results = []
        for entry in data:
            if "chosen" not in entry:
                raise ValueError(
                    "Preference entry missing 'chosen' field"
                )
            if "rejected" not in entry:
                raise ValueError(
                    "Preference entry missing 'rejected' field"
                )

            chosen = list(entry["chosen"])
            rejected = list(entry["rejected"])

            # Skip empty entries
            if not chosen or not rejected:
                continue

            # Add system prompt if present
            if "system" in entry and entry["system"]:
                system_msg = {"role": "system", "content": str(entry["system"])}
                chosen.insert(0, system_msg)
                rejected.insert(0, system_msg)

            # Validate: both must end with assistant
            chosen = self._trim_to_assistant(chosen)
            rejected = self._trim_to_assistant(rejected)

            if not chosen or not rejected:
                continue

            # Validate: prompts (all but last assistant message) must match
            chosen_prompt = chosen[:-1]
            rejected_prompt = rejected[:-1]

            if not self._prompts_match(chosen_prompt, rejected_prompt):
                continue

            # Validate: chosen != rejected (at least last message differs)
            if chosen[-1].get("content") == rejected[-1].get("content"):
                continue

            results.append({
                "chosen": chosen,
                "rejected": rejected,
            })

        return results

    def load_file(self, path: str) -> List[Dict]:
        """Load preference data from a JSON or JSONL file.

        Args:
            path: Path to the data file.

        Returns:
            List of validated preference pairs.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preference data file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        if not data:
            return []
        return self.load_standard(data)

    def compute_statistics(
        self,
        data: List[Dict],
        tokenizer=None,
    ) -> Dict[str, Any]:
        """Compute statistics for preference dataset.

        Args:
            data: List of preference pairs.
            tokenizer: Optional tokenizer for length estimation.

        Returns:
            Dict with statistics.
        """
        if not data:
            return {"num_pairs": 0}

        total_messages = 0
        total_chosen_len = 0
        total_rejected_len = 0

        for pair in data:
            chosen = pair.get("chosen", [])
            rejected = pair.get("rejected", [])
            total_messages += len(chosen) + len(rejected)

            for msg_list, counter in [(chosen, "chosen"), (rejected, "rejected")]:
                full_text = " ".join(
                    str(m.get("content", "")) for m in msg_list
                )
                if tokenizer and hasattr(tokenizer, "encode"):
                    tokens = len(tokenizer.encode(full_text))
                else:
                    tokens = len(full_text) // 2
                if counter == "chosen":
                    total_chosen_len += tokens
                else:
                    total_rejected_len += tokens

        n = len(data)
        return {
            "num_pairs": n,
            "total_messages": total_messages,
            "avg_messages_per_pair": round(total_messages / n, 2) if n else 0,
            "avg_chosen_length": total_chosen_len // n if n else 0,
            "avg_rejected_length": total_rejected_len // n if n else 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_assistant(messages: List[Dict]) -> List[Dict]:
        """Trim trailing non-assistant messages."""
        while messages and messages[-1].get("role") != "assistant":
            messages.pop()
        return messages

    @staticmethod
    def _prompts_match(
        prompt_a: List[Dict],
        prompt_b: List[Dict],
    ) -> bool:
        """Check if two prompts (message lists) are identical."""
        if len(prompt_a) != len(prompt_b):
            return False
        for ma, mb in zip(prompt_a, prompt_b):
            if ma.get("role") != mb.get("role"):
                return False
            if ma.get("content") != mb.get("content"):
                return False
        return True
