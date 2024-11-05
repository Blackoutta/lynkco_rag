from modelscope import snapshot_download

model_dir = snapshot_download(
    "Xorbits/bge-reranker-large", local_dir="./pretrained_models/bge-reranker-large"
)
print(f"下载完成, 模型保存在 {model_dir}")
