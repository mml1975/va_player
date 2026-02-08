# Test voice recognition and call player (without responce)

import os, re
import torch; torch._dynamo.config.recompile_limit = 64;

from unsloth import FastLanguageModel # Изменили импорт для работы с LoRA

import csv
import json
import random
import re
from typing import Dict, List, Optional
import requests
import time
import sys
from pathlib import Path

from tqdm import tqdm
from collections import namedtuple

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

from datasets import Audio
import numpy as np

base_dir = './audioplayer_va'
audioplayer_va = 'audioplayer_va.csv'
csvfilepath = Path(base_dir) / audioplayer_va
audiofiledir = Path(base_dir) / 'audio'

import pandas as pd
import os

df = pd.read_csv(
    csvfilepath,
    delimiter=",",
)
df.head(20)

def transcribe_audio(filename):
    """ Распознает аудио из filename и выдает запрос для audio_play, если надо."""
    SYSTEM_PROMPT_V1 = """You are a tool-using assistant that can understand audio. You are given a user's request in audio format.
You MUST transcribe user's audio and respond with exactly one XML tool call if the tool usage is required (user is asking about play music, asking music title or artist or Genre or Album), otherwise respond to user's request in plain text.
Template: <tool_call><name>audioplay.play_request</name><arguments><artist>ARTIST</artist><title>TITLE</title><genre>GENRE</genre><album>Album</album></arguments></tool_call>. If some arguments is not recognised the correpondence field should be empty. If tool usage is not required respond shortly with text

Also You MUST transcribe user's control command's to stop playing, such as stop, break, cancel to following Template: <tool_call><name>audioplay.play_stop</name><arguments></arguments></tool_call>

Also You MUST transcribe user's control command's about playing status, such as status (give status) to following Template: <tool_call><name>audioplay.play_status</name><arguments></arguments></tool_call>
"""

    messages = [            
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT_V1,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": filename},
                                
            ]
        }
    ]

    #print(messages)
    response = capture_inference(messages)
    return response
    
    
import csv

num = 0
responses=[]
for row in tqdm(df.itertuples()):
    num+=1
    #if num > 5:
    #    break    
    text_gtresponse = row.assistant_text # То, что ответила модель на просто текст
    user_audio = str(Path(audiofiledir) / row.user_audio) # Путь к wav

    #print(f'user_audio = {user_audio}')
    resp = transcribe_audio(user_audio)
    responses.append((row.user_text, row.user_audio, row.assistant_text, resp))

# Запишем csv

csv_filename = 'audioplayer_va_asr_fn.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['user_text', 'user_audio', 'assistant_text', 'asr_text'])
    for user_text, user_audio, assistant_text, asr_text in responses:
        writer.writerow([user_text, user_audio, assistant_text, asr_text])
        
print("CSV file created successfully")
    
    