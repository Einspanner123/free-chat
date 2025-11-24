from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    repo_type="model",
    local_dir="./Qwen/Qwen3-0.6B",
)
