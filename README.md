# 项目介绍

## 项目结构
```
.
├── 0-chunk_docs.py
├── 1-query_transformation.py
├── 2-predict.py
├── 3-evaluate.py
├── data
├── llm
├── pdf
├── retriever
├── pretrained_models
├── pytest.ini
├── requirements.txt
├── scratch.py
└── util.py
```
重要模块介绍:
- data: 存放各类输入输出文件及数据
- llm: vllm大模型模块
- pdf: pdf解析+切割模块
- retriever: 召回、reranker模块
- pretrained_models: 预训练模型下载脚本+存放目录

## 项目运行方法
依次运行:
- 0-chunk_docs.py
- 1-query_transformation.py
- 2-predict.py
- 3-evaluate.py

## 评估结果查看
3-evaluate.py会输出评估结果:
- ./data/metrics_detail.json # 评估详情
- ./data/metrics_mean.json   # 评估总分