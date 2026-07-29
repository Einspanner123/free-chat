from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FilterConfig:
    min_length: int = 10
    max_length: int = 2048
    deduplicate: bool = True
    remove_html: bool = True
    max_repetition_ratio: float = 0.5
    max_input_output_overlap: float = 0.8

@dataclass
class SynthConfig:
    num_seed_examples: int = 50
    max_generated: int = 10000
    temperature: float = 0.8
    top_p: float = 0.95
    strategies: List[str] = field(default_factory=lambda: ["self_instruct", "evol_question"])
    filter: FilterConfig = field(default_factory=FilterConfig)
    output_dir: str = "./synthetic_data"
