import torch
import sounddevice as sd
import wavio
import numpy as np
from silero_vad import load_silero_vad
import time

# Параметры аудио
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Фиксированный размер для модели при 16 кГц
CHUNK_DURATION_MS = int((CHUNK_SIZE / SAMPLE_RATE) * 1000)

# Параметры детектирования конца реплики
PAUSE_THRESHOLD_MS = 500
PAUSE_CHUNKS = PAUSE_THRESHOLD_MS // CHUNK_DURATION_MS

# Загружаем модель и переводим в режим инференса
model = load_silero_vad()
model.eval()

# Состояние записи
audio_buffer = []
is_recording = False
pause_counter = 0

def callback(indata, frames, time_info, status):
    global audio_buffer, is_recording, pause_counter
    
    if status:
        print(f"Аудиостатус: {status}")
    
    # Конвертируем в моно и выравниваем размер
    if indata.ndim > 1:
        audio_chunk = indata.mean(axis=1).flatten()
    else:
        audio_chunk = indata.flatten()
    
    # Критически важно: размер чанка должен быть ровно 512 семплов
    if len(audio_chunk) > CHUNK_SIZE:
        audio_chunk = audio_chunk[:CHUNK_SIZE]
    elif len(audio_chunk) < CHUNK_SIZE:
        # Дополняем нулями (тишиной) до нужного размера
        padding = np.zeros(CHUNK_SIZE - len(audio_chunk))
        audio_chunk = np.concatenate([audio_chunk, padding])
    
    # Преобразуем в тензор PyTorch и получаем вероятность речи
    audio_tensor = torch.from_numpy(audio_chunk).float()
    
    with torch.no_grad():
        # Модель ожидает вход [batch_size, samples]
        # unsqueeze(0) добавляет размерность батча
        prob_tensor = model(audio_tensor.unsqueeze(0), SAMPLE_RATE)
        
        # Для отладки можно раскомментировать:
        # print(f"Форма вероятностей: {prob_tensor.shape}, значения: {prob_tensor}")
        
        # Получаем скалярное значение вероятности
        # Если prob_tensor имеет форму [1, 1], используем .item()
        # Если форма [1, n], нужно агрегировать (например, взять среднее)
        if prob_tensor.numel() == 1:
            speech_prob = prob_tensor.item()
        else:
            # Берем среднее по временным фреймам
            speech_prob = prob_tensor.mean().item()
    
    # Логика детектирования речи и записи
    if speech_prob > 0.5:  # Порог можно настроить (обычно 0.5)
        is_recording = True
        pause_counter = 0
        audio_buffer.append(audio_chunk.copy())
        print("🎤 Речь обнаружена", end='\r')
    elif is_recording:
        pause_counter += 1
        audio_buffer.append(audio_chunk.copy())
        
        if pause_counter >= PAUSE_CHUNKS:
            # Сохраняем записанную реплику
            if audio_buffer:
                full_audio = np.concatenate(audio_buffer, axis=0)
                timestamp = int(time.time() * 1000)
                filename = f"utterance_{timestamp}.wav"
                wavio.write(filename, full_audio, SAMPLE_RATE, sampwidth=2)
                duration = len(full_audio) / SAMPLE_RATE
                print(f"\n✅ Реплика сохранена: {filename} ({duration:.2f} сек)")
                
                # Сбрасываем состояние для следующей реплики
                audio_buffer.clear()
                is_recording = False
                pause_counter = 0

# Настройка и запуск аудиопотока
print(f"Конфигурация: Частота {SAMPLE_RATE} Гц, чанк {CHUNK_SIZE} семплов")
print(f"Порог паузы: {PAUSE_THRESHOLD_MS} мс")

try:
    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=callback,
        dtype='float32'
    )
    
    stream.start()
    print("\n🎤 Система готова. Говорите... (Ctrl+C для остановки)")
    print("Индикация: '🎤 Речь обнаружена' при активной речи")
    
    # Бесконечный цикл ожидания
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\nЗапись остановлена пользователем.")
except Exception as e:
    print(f"\nОшибка: {e}")
finally:
    if 'stream' in locals():
        stream.stop()
        stream.close()
    
    # Сохраняем последнюю реплику, если запись была прервана во время речи
    if audio_buffer and is_recording:
        full_audio = np.concatenate(audio_buffer, axis=0)
        filename = f"utterance_final_{int(time.time()*1000)}.wav"
        wavio.write(filename, full_audio, SAMPLE_RATE, sampwidth=2)
        print(f"Сохранена последняя реплика: {filename}")
