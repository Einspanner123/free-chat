import os
import json
from pathlib import Path
from threading import Thread

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

os.environ["HF_HUB_OFFLINE"] = "1"


class ChatModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        repeat_penalty: float = 1.05,
        top_p: float = 0.7,
        top_k: float = 40,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.top_p = top_p
        self.top_k = top_k
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        model_path = Path(__file__).parent / "model" / model_name
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,  # Qwen模型需要这个参数
            local_files_only=True,  # 仅从本地加载，不尝试下载
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype="auto",  # 自动选择数据类型
        ).to(self.device)

    def GetStreamer(self, msg):
        try:
            # 尝试解析为JSON消息列表
            messages = json.loads(msg)
            if isinstance(messages, list):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = msg
        except (json.JSONDecodeError, TypeError):
            # 解析失败，回退到原始字符串
            text = msg

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repeat_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        return streamer
