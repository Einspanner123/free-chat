"""Conservative per-rank feasibility, separate from dynamic request reservation."""

from freechat_contracts import ModelCapability, RequestProfile


def required_kv_bytes_per_rank(request: RequestProfile, model: ModelCapability) -> int | None:
    geometry = model.kv_admission_bytes_per_token_per_rank
    block = model.kv_block_size_tokens
    if geometry is None or block is None:
        return None
    tokens = request.input_tokens + request.output_tokens
    aligned_tokens = ((tokens + block - 1) // block) * block
    # Do not credit unproven prefix residency or divide MLA/GQA replication by TP.
    return aligned_tokens * geometry
