import json


def load_test_questions(file_path: str, n: int = -1, start=0):
    with open(file_path, "r", encoding="utf-8") as file:
        question_list = json.load(file)
        question_list = [question["question"] for question in question_list]
    print(f"已加载{len(question_list)} 条问题")
    if n != -1:
        return question_list[start : start + n]
    return question_list[start:]


def load_test_questions_rewritten(file_path: str, n: int = -1, start=0):
    with open(file_path, "r", encoding="utf-8") as file:
        question_list = json.load(file)
        question_list = [question["concat"] for question in question_list]
    print(f"已加载{len(question_list)} 条问题")
    if n != -1:
        return question_list[start : start + n]
    return question_list[start:]
