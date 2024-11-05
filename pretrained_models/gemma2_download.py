from modelscope import snapshot_download

model_dir = snapshot_download(
    "LLM-Research/gemma-2-2b-it", local_dir="./pretrained_models/gemma-2-2b-it"
)
print(f"下载完成, 模型保存在 {model_dir}")
