from langchain.schema import Document
import jieba
from typing import List
from langchain_community.retrievers import BM25Retriever


class BM25Recaller(object):
    token_docs: List[Document] = []
    full_docs: List[Document] = []
    retriever: BM25Retriever

    def __init__(self, strs: List[str]) -> None:
        for idx, str in enumerate(strs):
            # 清除换行符和两边的空格
            str = str.strip("\n").strip()
            if len(str) < 5:
                continue
            # 分词后录入为文档
            tokens = " ".join(jieba.cut_for_search(str))
            self.token_docs.append(Document(page_content=tokens, metadata={"id": idx}))

            # 完整文档录入
            words = str.split("\t")
            self.full_docs.append(Document(page_content=words[0], metadata={"id": idx}))

        self.retriever = BM25Retriever.from_documents(self.token_docs)

    def topk(self, query, k) -> List[Document]:
        self.retriever.k = k
        query = " ".join(jieba.cut_for_search(query))
        matches = self.retriever.get_relevant_documents(query)
        full_docs = []
        for match in matches:
            full_docs.append(self.full_docs[match.metadata["id"]])
        return full_docs


def test_bm25_retriever():
    strs = ["这是第一行", "这是第二行", "这是第三行"]
    bm25 = BM25Recaller(strs)

    docs = bm25.topk(query="第二行", k=1)
    for doc in docs:
        print(doc.page_content)

    assert len(docs) == 1
    assert docs[0].page_content == "这是第二行"
