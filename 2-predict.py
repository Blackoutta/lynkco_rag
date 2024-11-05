import os
from pdf import parse_pdf
from retriever import FaissRecaller, BM25Recaller, Reranker
from llm import ChatLLM
import json
from typing import List, Dict
from langchain.schema import Document
import time
from util import *


def main():
    """
    对重写后的测试问题，进行批量线下推理，并生成推理结果记录文件
    输入:
    - ./data/parsed/test_questions_rewritten.json
    - ./data/parsed/all_text.txt
    输出:
    - ./data/submission.json

    """
    model_base = "./pretrained_models"
    llm_model_name = "qwen-3b-instruct-gptq-int8"
    llm_model_path = os.path.join(model_base, llm_model_name)

    vectorstore_path = "./data/vectorstore"
    embedding_model_name = "m3e-large"
    embedding_model_path = os.path.join(model_base, embedding_model_name)

    reranker_model_name = "bge-reranker-large"
    reranker_model_path = os.path.join(model_base, reranker_model_name)

    # 读取数据
    data_path = "./data/parsed/all_text.txt"
    if not os.path.exists(data_path):
        print("数据文件不存在，正在解析pdf文件...")
        parse_pdf()

    print("数据文件存在，正在读取数据...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    print(f"已加载{len(data)} 条数据")

    # Faiss
    if not os.path.exists(os.path.join(vectorstore_path, "lynkco.faiss")):
        print(f"向量数据库不存在，正在生成向量文件并保存至本地: {vectorstore_path}...")
        faiss_recaller = FaissRecaller(
            embedding_model_path=embedding_model_path, data=data
        )
        faiss_recaller.vector_store.save_local(
            folder_path=vectorstore_path, index_name="lynkco"
        )
        print("向量数据库已保存至本地!")
    else:
        print("向量数据库已存在，正在加载向量文件...")
        faiss_recaller = FaissRecaller.from_local(
            embedding_model_path=embedding_model_path,
            folder_path=vectorstore_path,
            index="lynkco",
        )
    print("Faiss数据已加载!")

    # BM25
    bm25_recaller = BM25Recaller(strs=data)
    print("BM25数据已加载!")

    # Reranker
    reranker = Reranker(model_path=reranker_model_path)
    print("Reranker已加载!")

    # 大模型
    llm = ChatLLM(model_path=llm_model_path)
    print("大模型已加载!")

    # 加载测试问题集
    # questions = load_test_questions(file_path="./data/test_question.json")
    questions = load_test_questions_rewritten(
        file_path="./data/test_question_rewritten.json", start=0, n=-1
    )

    submissions = []
    prompt_batches = []

    start = time.time()
    for idx, question in enumerate(questions):
        faiss_docs = []
        bm25_contexts = []
        reranked_contexts = []
        qs = question.split("\n")
        original_question = qs[0]
        for q in qs:
            # faiss召回
            faiss_contexts = faiss_recaller.topk(q, k=15)
            fc = [doc for doc, score in faiss_contexts]
            faiss_docs.extend(fc)

            # bm25召回
            bc = bm25_recaller.topk(query=q, k=15)
            bm25_contexts.extend(bc)

            # 多路召回rerank
            rc = reranker.do_rerank(query=q, docs=fc + bc)
            reranked_contexts.extend(rc)

        # 基于召回文档构建上下文string
        emb_candidates = sample_and_merge_candidates([faiss_docs], max_length=4000)
        bm25_candidates = sample_and_merge_candidates([bm25_contexts], max_length=4000)
        hybrid_candidates = sample_and_merge_candidates(
            [faiss_docs, bm25_contexts], max_length=2500
        )
        reranked_candidates = sample_and_merge_candidates(
            [reranked_contexts], max_length=5000
        )

        # 清洗上下文，这里忽略了，提升效果不大
        # ctx_summary = llm.batch_infer(
        #     [
        #         compose_summary_prompt(ctxs=emb_candidates),
        #         compose_summary_prompt(ctxs=bm25_candidates),
        #         compose_summary_prompt(ctxs=hybrid_candidates),
        #         compose_summary_prompt(ctxs=reranked_candidates),
        #     ]
        # )

        # 构建最终给到大模型的prompts
        emb_prompt = compose_rag_prompt(ctxs=emb_candidates, query=original_question)
        bm25_prompt = compose_rag_prompt(ctxs=bm25_candidates, query=original_question)
        hybrid_prompt = compose_rag_prompt(
            ctxs=hybrid_candidates, query=original_question
        )
        reranked_prompt = compose_rag_prompt(
            ctxs=reranked_candidates, query=original_question
        )
        print(
            f"构建prompt完成! emb: {len(emb_prompt)} 字符, bm25: {len(bm25_prompt)} 字符, hybrid: {len(hybrid_prompt)} 字符, reranked: {len(reranked_prompt)} 字符"
        )

        prompt_batch = [bm25_prompt, emb_prompt, hybrid_prompt, reranked_prompt]
        prompt_batches.append(prompt_batch)

    # 释放无用显存
    reranker.release()
    faiss_recaller.release()
    print("reranker, faiss recaller已释放资源!")

    # 开始批量推理
    flatten_prompt_batches = [prompt for batch in prompt_batches for prompt in batch]
    llm_answers = llm.batch_infer(prompts=flatten_prompt_batches)

    answer_batches = [llm_answers[i : i + 4] for i in range(0, len(llm_answers), 4)]

    for question, prompt_batch, answer_batch in zip(
        questions, prompt_batches, answer_batches
    ):
        submission = {
            "question": question,
            "bm25": {"prompt": prompt_batch[0], "answer": answer_batch[0]},
            "emb": {"prompt": prompt_batch[1], "answer": answer_batch[1]},
            "hybrid": {"prompt": prompt_batch[2], "answer": answer_batch[2]},
            "reranked": {"prompt": prompt_batch[3], "answer": answer_batch[3]},
        }
        submissions.append(submission)

    # 保存答案生成结果
    json.dump(
        submissions,
        open("./data/submission.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    print(f"已保存{len(submissions)} 条答案生成结果!")

    end = time.time()

    print(f"总召回+推理耗时: {end - start:.2f} 秒")


def sample_and_merge_candidates(
    sources: List[List[Document]], max_length=2500, max_cnt=6
) -> List[str]:
    answers = []
    for docs in sources:
        sorted_list = sorted(
            docs,
            reverse=True,
            key=lambda x: (
                x.metadata["score"]
                if x.metadata["reranked_score"] is None
                else x.metadata["reranked_score"]
            ),
        )
        clean_docs = remove_dups(sorted_list)
        cnt = 0
        answer = ""
        for i, doc in enumerate(clean_docs):
            cnt = cnt + 1
            if len(answer + doc) > max_length:
                break
            if doc in answer:
                continue
            answer = answer + doc + "\n"
            # 最多选6个
            if cnt > max_cnt:
                break

        answers.append(answer)
    return answers


def remove_dups(docs: List[Document]):
    dup_dict = {}
    clean_docs: List[str] = []
    for doc in docs:
        id = doc.metadata["id"]
        if id not in dup_dict:
            clean_docs.append(doc.page_content)
            dup_dict[id] = 1
    return clean_docs


def compose_rag_prompt(ctxs: List[str], query):
    """
    构建RAG prompt
    """
    final_ctx = "\n".join(ctxs)

    prompt_template = f"""
    你需要基于已知上下文信息，简洁、不重复和专业的来回答我的问题。
    如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加上下文中没有的编造成分，答案请使用中文。
    # 上下文:
    ```
    {final_ctx}
    ```
    # 我的问题:
    ```
    {query}
    ```
    请按Markdown格式输出, 分成不同点来回答。
    """
    return prompt_template


def compose_summary_prompt(ctxs: List[str]):
    ctx = "\n".join(ctxs)
    return f"""
    # 任务
    请以专业汽车售后人员的方式，整理并总结以下内容，要求:
    - 简洁明了，不重复
    - 不遗漏重要信息

    # 目标内容
    ```
    {ctx}
    ```

    # 输出格式
    用Markdown格式分点输出，格式示例:
    # 上下文分点总结
    1. xxx
    2. xxx
    3. xxx
    ...
    """


if __name__ == "__main__":
    main()
