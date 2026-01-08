from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    allow_patterns="libero_90/*",
    local_dir="/data/yihong.ji/RobustVLA-283D/LIBERO/libero/datasets",
)
