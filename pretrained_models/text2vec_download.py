from modelscope import snapshot_download

model_dir = snapshot_download(
    "Jerry0/text2vec-base-chinese",
    local_dir="./pretrained_models/text2vec-base-chinese",
)
print(f"下载完成, 模型保存在 {model_dir}")
