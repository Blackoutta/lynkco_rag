# coding: utf-8


from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
from typing import List
import os


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class FaissRecaller(object):
    embeddings: HuggingFaceEmbeddings
    docs: List[Document] = []
    vectorStore: FAISS

    def __init__(self, embedding_model_path: str, data: List[str]):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path, model_kwargs={"device": "cuda"}
        )
        if data is not None:
            docs = []
            for idx, line in enumerate(data):
                line = line.strip("\n").strip()
                words = line.split("\t")
                docs.append(Document(page_content=words[0], metadata={"id": idx}))
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            del self.embeddings
            torch.cuda.empty_cache()

    @staticmethod
    def from_local(embedding_model_path: str, folder_path: str, index="lynkco"):
        instance = FaissRecaller(embedding_model_path=embedding_model_path, data=None)
        vector_store = FAISS.load_local(
            folder_path=folder_path,
            index_name=index,
            embeddings=instance.embeddings,
            allow_dangerous_deserialization=True,
        )
        instance.vector_store = vector_store
        return instance

    # 获取top-K分数最高的文档块
    def topk(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store

    def release(self):
        del self.embeddings
        torch_gc()


def test_faiss_retriever():
    model_path = os.path.join("pretrained_models", "m3e-large")
    data = open("data/parsed/block_512.txt", "r", encoding="utf-8").readlines()
    faiss_retriever = FaissRecaller(embedding_model_path=model_path, data=data)

    founds = faiss_retriever.topk("吉利汽车语音组手叫什么", 1)
    for found in founds:
        print(found[0].page_content, found[1])
