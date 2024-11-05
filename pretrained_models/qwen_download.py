from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct", local_dir="./pretrained_models/qwen-3b-instruct"
)

snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
    local_dir="./pretrained_models/qwen-3b-instruct-gptq-int8",
)
