# Test text API for audioplay tool

import os, re
import torch; torch._dynamo.config.recompile_limit = 64;

# from unsloth import FastModel
from unsloth import FastLanguageModel # Изменили импорт для работы с LoRA

"""
model, processor = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)
"""

model, processor = FastLanguageModel.from_pretrained(
    model_name = "gemma3n_audioplayer_lora", # Указываем путь к сохраненным LoRA адаптерам
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    # full_finetuning = False, # Этот параметр больше не нужен при загрузке LoRA
)

# функции для инференса

from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template


# Check out available chat templates and how they are applied
# https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py
tokenizer = get_chat_template(processor, chat_template="gemma3n")

# Helper function for inference
def streaming_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        do_sample=False,
        streamer = TextStreamer(processor, skip_prompt = True),
    )
    
from transformers import TextStreamer
import io
import sys


class CaptureTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Сохраняем текст
        self.generated_text += text
        # Также выводим в консоль (опционально)
        print(text, end="", flush=True)

# Модифицированная функция для захвата текста
def capture_inference(messages, max_new_tokens=128):
    # Создаем кастомный streamer
    streamer = CaptureTextStreamer(processor, skip_prompt=True)
    
    # Генерируем ответ
    output = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )
    
    # Возвращаем захваченный текст
    return streamer.generated_text

from datasets import Dataset, Audio, Features, Value
from pathlib import Path
import pandas as pd

data_path = "audio_assistant_queries.csv"    

df = pd.read_csv(
    data_path,
    delimiter=",",
)
df.head(15)


from tqdm import tqdm
from collections import namedtuple

# Определяем именованную структуру
ResponseRecord = namedtuple('ResponseRecord', ['request', 'ground_truth', 'real_response'])

SYSTEM_PROMPT = """You are a tool-using assistant that can understand audio. You are given a user's request in audio format.
You MUST transcribe user's audio and respond with exactly one XML tool call if the tool usage is required (user is asking about play music, asking music title or artist or Genre or Album), otherwise respond to user's request in plain text.
Template: <tool_call><name>audioplay.play_request</name><arguments><artist>ARTIST</artist><title>TITLE</title><genre>GENRE</genre><album>Album</album></arguments></tool_call>. If some arguments is not recognised the correpondence field should be empty. If tool usage is not required respond shortly with text

Also You MUST transcribe user's control command's to stop playing, such as stop, break, cancel to following Template: <tool_call><name>audioplay.play_stop</name><arguments></arguments></tool_call>

Also You MUST transcribe user's control command's about playing status, such as status (give status) to following Template: <tool_call><name>audioplay.play_status</name><arguments></arguments></tool_call>
"""

result = []

for row in tqdm(df.itertuples()):
    text_request = row.Request
    text_gtresponse = row.Response
    
    test_messages = [            
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [            
                {"type": "text", "text": text_request}
                
            ]
        }
    ]
    
    # Получаем реальный ответ модели
    print(f"request: {text_request}\n")
    real_response = capture_inference(test_messages)
    print()
    print(f"\nground truth: {text_gtresponse}\n")
    
    # Создаем именованную запись
    record = ResponseRecord(
        request=text_request,
        ground_truth=text_gtresponse,
        real_response=real_response
    )
    
    result.append(record)
    
# Сохраняем результат в JSON файл
# Преобразуем namedtuple в словарь для сериализации
import json
result_dicts = [record._asdict() for record in result]

with open('gemma_audio_va_results3fn.json', 'w', encoding='utf-8') as f:
    json.dump(result_dicts, f, ensure_ascii=False, indent=2)