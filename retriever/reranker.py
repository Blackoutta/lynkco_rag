from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import os
import torch
from typing import List, Union
from langchain.schema import Document
import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"


# 释放gpu显存及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class Reranker(object):
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    model: any
    max_length: int

    def __init__(self, model_path: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.cuda()
        self.max_length = max_length

    def do_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(device)
        with torch.inference_mode():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()

        response = []
        sorted_list = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])
        for score, doc in sorted_list:
            doc.metadata["reranked_score"] = score.item()
            response.append(doc)

        torch_gc()
        return response

    def release(self):
        del self.model
        torch_gc()


def test_reranker():
    reranker = Reranker(
        model_path="./pretrained_models/bge-reranker-large", max_length=512
    )

    preds = reranker.do_rerank(
        query="银渐层掉毛量多么？",
        docs=[
            Document(page_content="银渐层掉毛量很多, 是猫咪中掉毛量最大的品种之一。"),
            Document(page_content="银渐层猫咪很可爱哟～"),
            Document(page_content="布偶猫很可爱，掉毛量也很少。"),
            Document(
                page_content="边境牧羊犬是牧羊犬的一种, 有着非常聪明的智商和活泼好动的性格。"
            ),
            Document(
                page_content="金毛犬是犬类中非常受欢迎的品种之一, 有着非常聪明和友善的性格。"
            ),
        ],
    )
    for doc in preds:
        print(doc.page_content, doc.metadata["score"])

    reranker.release()
