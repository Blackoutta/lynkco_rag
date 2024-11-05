from modelscope import snapshot_download

model_dir = snapshot_download(
    "AI-ModelScope/m3e-large", local_dir="./pretrained_models/m3e-large"
)
print(f"下载完成, 模型保存在 {model_dir}")
