import torch
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    GenerationConfig,
)
from transformers import GenerationConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
import time
import os


class ChatLLM(ABC):
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    generation_config: GenerationConfig
    stop_words_ids: List[int] = []
    model: LLM

    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        # 加载vLLM大模型
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
        )

        self.sampling_params = SamplingParams(
            top_p=1.0,
            top_k=(
                -1
                if self.generation_config.top_k == 0
                else self.generation_config.top_k
            ),
            temperature=0.1,
            max_tokens=2000,
            repetition_penalty=self.generation_config.repetition_penalty,
            n=1,
        )
        # self.beam_search_params = BeamSearchParams(
        #     beam_width=1,
        #     max_tokens=2000,
        #     temperature=0.3,
        #     length_penalty=self.generation_config.repetition_penalty,
        # )

    def batch_infer(self, prompts: List[str]):
        # 准备输入
        batch_text = []
        for p in prompts:
            messages = [
                {"role": "user", "content": p},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_text.append(text)

        # 开始推理
        # sampling
        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)

        # beam search
        # outputs = self.model.beam_search(
        #     prompts=batch_text, params=self.beam_search_params
        # )

        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            batch_response.append(output_str)
        torch_gc()
        return batch_response


# 释放gpu显存及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


"""
Testing
"""


def test_chatllm():
    base_path = "./pretrained_models"
    model_name = "qwen-3b-instruct-gptq-int8"
    # model_name = "gemma-2-2b-it"

    llm = ChatLLM(model_path=os.path.join(base_path, model_name))
    prompts = ["汽车的部件有哪些？", "什么是自动驾驶？", "世界上最好的汽车是什么？"]
    start = time.time()
    answers = llm.batch_infer(prompts)
    end = time.time()
    for prompt, answer in zip(prompts, answers):
        print(f"prompt: {prompt}")
        print(f"answer: {answer}")
        print("-" * 100)

    print(f"cost time: {end - start:.2f} seconds")


if __name__ == "__main__":
    test_chatllm()
