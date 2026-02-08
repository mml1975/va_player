import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import re

# Загрузка данных
data_path = "audioplayer_va.csv"
df = pd.read_csv(data_path, delimiter=",")
print(f"Загружено {len(df)} примеров")

# Подготовка данных для обучения
def prepare_dataset(df):
    """
    Подготавливает датасет в формате для обучения с чат-шаблоном.
    """
    conversations = []
    
    for idx, row in df.iterrows():
        user_text = row['user_text']
        assistant_text = row['assistant_text']
        
        # Очищаем assistant_text от <end_of_turn> если он есть
        cleaned_assistant_text = re.sub(r'\s*<end_of_turn>\s*', '', assistant_text).strip()
        
        # Создаем диалог в формате, подходящем для чат-шаблона
        conversation = [
            {
                "role": "user",
                "content": user_text
            },
            {
                "role": "assistant",
                "content": cleaned_assistant_text
            }
        ]
        conversations.append(conversation)
    
    return Dataset.from_list([{"messages": conv} for conv in conversations])

dataset = prepare_dataset(df)

# Загрузка модели
max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True  # Используем 4-bit квантование

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Устанавливаем чат-шаблон
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma3n",  # Gemma-3n использует специфичный шаблон
)

# Применяем чат-шаблон к модели
FastLanguageModel.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj",]

# Настройка LoRA адаптеров
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Ранг LoRA - можно увеличить для лучшей производительности
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Экономия памяти
    random_state=3407,
    use_rslora=False,  # Используем стандартный LoRA
    loftq_config=None,  # LoftQ не нужен для квантованных моделей
)

# Функция форматирования примеров с использованием чат-шаблона
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Применяем форматирование
dataset = dataset.map(formatting_prompts_func, batched=True)

# Настройка параметров обучения
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Уменьшите, если мало VRAM
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=500,  # Настройте количество шагов под ваш датасет
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",  # Отключаем MLflow и TensorBoard
)

# Создание тренера
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Отключаем packing для более точного контроля
    args=training_args,
    formatting_func=formatting_prompts_func,
)

# Обучение
trainer_stats = trainer.train()

# Сохранение модели
model.save_pretrained("gemma3n_audioplayer_lora")  # Только LoRA адаптеры
tokenizer.save_pretrained("gemma3n_audioplayer_lora")

print("Файнтюнинг завершен!")
print(f"Итоговая потеря: {trainer_stats.training_loss}")