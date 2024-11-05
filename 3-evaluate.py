from typing import List, Dict
import numpy as np
from text2vec import SentenceModel, semantic_search
import torch
import json


def main():
    """
    对推理结果进行评估,输出评估记录文件
    输入:
    - ./data/gold.json
    - ./data/submission.json

    输出:
    - ./data/metrics_detail.json    # 得分详情
    - ./data/metrics_mean.json      # 平均总分

    """
    # 相似度模型
    semantic_model_path = "./pretrained_models/text2vec-base-chinese"
    semantic_model = SentenceModel(
        model_name_or_path=semantic_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # 标准答案路径
    gold_path = "./data/gold.json"
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = "./data/submission.json"
    print("Read predict file from %s" % predict_path)

    with open(gold_path, "r", encoding="utf-8") as f:
        golds = json.load(f)
    with open(predict_path, "r", encoding="utf-8") as f:
        predicts = json.load(open(predict_path))

    results = []
    for gold, pred in zip(golds, predicts):
        question = gold["question"].strip()
        gold_keywords = gold["keywords"]
        gold_answer = gold["answer"].strip()

        def calc_score(elem: Dict) -> float:
            pred_answer = elem["answer"]
            pred_keywords = [word for word in gold_keywords if word in pred_answer]
            if gold_answer == "无答案" and gold_answer not in pred_answer:
                return {
                    "answer": pred_answer,
                    "gold": gold_answer,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "final_score": 0.0,
                }
            if gold_answer == "无答案" and gold_answer in pred_answer:
                return {
                    "answer": pred_answer,
                    "gold": gold_answer,
                    "semantic_score": 1.0,
                    "keyword_score": 1.0,
                    "final_score": 1.0,
                }
            semantic_score = semantic_search(
                query_embeddings=semantic_model.encode(pred_answer),
                corpus_embeddings=semantic_model.encode(gold_answer),
                top_k=1,
            )[0][0]["score"]
            keyword_score = calc_jaccard(
                pred_keywords=pred_keywords,
                gold_keywords=gold_keywords,
            )

            return {
                "answer": pred_answer,
                "gold": gold_answer,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "final_score": 0.3 * keyword_score + 0.7 * semantic_score,
            }

        results.append(
            {
                "question": question,
                "bm25": calc_score(pred["bm25"]),
                "emb": calc_score(pred["emb"]),
                "hybrid": calc_score(pred["hybrid"]),
                "reranked": calc_score(pred["reranked"]),
            }
        )

    # 记录得分详情
    json.dump(
        results,
        open("./data/metrics_detail.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    bm25_mean_score = np.mean([result["bm25"]["final_score"] for result in results])
    emb_mean_score = np.mean([result["emb"]["final_score"] for result in results])
    hybrid_mean_score = np.mean([result["hybrid"]["final_score"] for result in results])
    reranked_mean_score = np.mean(
        [result["reranked"]["final_score"] for result in results]
    )

    # 记录总平均得分
    json.dump(
        {
            "bm25": bm25_mean_score,
            "emb": emb_mean_score,
            "hybrid": hybrid_mean_score,
            "reranked": reranked_mean_score,
        },
        open("./data/metrics_mean.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )
    print("done!")


def calc_jaccard(pred_keywords: List[str], gold_keywords: List[str], threshold=0.9):
    """
    Jaccard 相似系数（Jaccard Index），也叫做 Jaccard 相似度，是一种用于衡量两个集合相似度的指标。
    它的定义是两个集合交集的大小除以它们并集的大小。
    """
    union = [i for i in pred_keywords if i in gold_keywords]
    # 这里加上 1e-6 是为了防止当 size_b 为零时出现除零错误。
    score = len(union) / (len(pred_keywords) + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


if __name__ == "__main__":
    main()
