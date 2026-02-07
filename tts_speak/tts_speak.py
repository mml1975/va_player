import sys
import os
import sounddevice as sd
import numpy as np
import torch

# Сохраняем оригинальную функцию torch.load
original_torch_load = torch.load

# Создаем новую функцию, которая устанавливает weights_only=False
def custom_torch_load(*args, **kwargs):
    # Устанавливаем weights_only=False
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Заменяем torch.load на нашу функцию
torch.load = custom_torch_load

# Перед импортом любых библиотек ML
os.environ.update({
    'MIOPEN_FIND_ENFORCE': '3',
    'MIOPEN_DISABLE_WORKSPACE_FALLBACK': '0',
    'MIOPEN_WORKSPACE_LIMIT': '1073741824',
    'MIOPEN_DEBUG_CONV_GEMM': '0',
    'MIOPEN_LOG_LEVEL': '0',
    'PYTORCH_MIOPEN_CACHE': '1',
    'PYTORCH_ROCM_ARCH': 'gfx1100',
})

torch.cuda.empty_cache()

# Настроить backend для сверточных операций
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Для Coqui TTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

def load_tts_model():
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    return tts  # <-- ДОБАВИТЬ ЭТУ СТРОКУ!

def process_text(tts, spkfile, text):
    print("Генерация...")
    try:
        # Получаем частоту дискретизации из модели
        sample_rate = tts.synthesizer.output_sample_rate
        
        lang = 'en'
        audio_data = tts.tts(
            text=text,
            speaker_wav=spkfile,
            language=lang,
            split_sentences=True                        
        )
        
        print("Воспроизведение...")
        audio_array = np.array(audio_data)  # Преобразуем в numpy массив
        sd.play(audio_array, sample_rate)
        sd.wait()  # Ждем окончания воспроизведения
        sd.stop()
            
    except Exception as e:
        print(f"Ошибка {e}")
        
        

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python script.py <имя_файла голоса> <текст>")
    else:
        spkfile = sys.argv[1]
        text = sys.argv[2]
        if not os.path.exists(spkfile):
            print(f"Файл голоса не найден: {spkfile}")
            sys.exit(1)
        tts = load_tts_model()  # Теперь tts будет содержать объект модели
        process_text(tts, spkfile, text)