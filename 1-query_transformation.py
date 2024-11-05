from llm import ChatLLM
from typing import List, Dict
import os
from util import load_test_questions
import json


def main():
    """
    对测试问题进行预先重写，加速测试
    输入:./data/test_question.json
    输出: ./data/test_question_rewritten.json
    """
    model_base = "./pretrained_models"
    llm_model_name = "qwen-3b-instruct-gptq-int8"
    llm_model_path = os.path.join(model_base, llm_model_name)

    questions = load_test_questions(
        file_path="./data/test_question.json", start=0, n=-1
    )
    prompts = [compose_prompt(elem) for elem in questions]

    # 大模型
    llm = ChatLLM(model_path=llm_model_path)
    print("大模型已加载!")

    batch_output = llm.batch_infer(prompts=prompts)

    results = []
    for original_question, rewritten_question in zip(questions, batch_output):
        e = {
            "original": original_question,
            "rewritten": rewritten_question,
            "concat": "-" + original_question + "\n" + rewritten_question,
        }
        results.append(e)

    json.dump(
        results,
        open("./data/test_question_rewritten.json", "w"),
        ensure_ascii=False,
        indent=4,
    )


def compose_prompt(q):
    prompt_template = """
    # 角色与背景
    你是领克汽车的专业售后人员，我是你的客户。

    # 任务
    我会问你一个问题，而你需要理解、分析我的问题，并将我的问题改写为3个语义相似的问题。

    # 我的问题
    {question}

    # 输出格式
    请用markdown格式输出，严格遵照以下格式，不要输出与我的问题无关的内容:
    - xxx
    - xxx
    - xxx
    """.format(
        question=q
    )
    return prompt_template


if __name__ == "__main__":
    main()
