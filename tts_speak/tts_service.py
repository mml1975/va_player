#!/usr/bin/env python3
"""
Микросервис TTS (Text-to-Speech) для генерации и воспроизведения речи
"""

import sys
import os
import threading
import time
import tempfile
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import logging
import numpy as np
import torch

# ========== Настройки для Coqui TTS ==========
# Сохраняем оригинальную функцию torch.load
original_torch_load = torch.load

# Создаем новую функцию, которая устанавливает weights_only=False
def custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Заменяем torch.load на нашу функцию
torch.load = custom_torch_load

# Настройки окружения для ML
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
os.environ["COQUI_TOS_AGREED"] = "1"

# ========== Импорт библиотек после настроек ==========
import sounddevice as sd
from TTS.api import TTS
import pygame
from typing import Optional, Dict, Any

# ========== Настройка логгирования ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ========== Глобальные переменные ==========
# Инициализируем конфиг пустым словарем, он будет заполнен позже
config: Dict[str, Any] = {}
# Голос по умолчанию будет установлен после загрузки конфига
DEFAULT_SPEAKER = None

tts_model = None
current_audio_data: Optional[np.ndarray] = None
is_playing = False
stop_event = threading.Event()
player_thread: Optional[threading.Thread] = None
sample_rate = 24000  # Частота дискретизации XTTS по умолчанию

# ========== Функции загрузки конфигурации ==========
def load_config(config_path: str = "tts_config.json") -> Dict[str, Any]:
    """Загрузка конфигурации из JSON файла"""
    config_file = Path(config_path)
    loaded_config = {}
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            logger.info(f"Конфигурация загружена из {config_file}")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
    else:
        logger.warning(f"Файл конфигурации не найден: {config_file}. Используются значения по умолчанию.")
        # Создаем файл конфигурации с настройками по умолчанию
        loaded_config = {
            "host": "127.0.0.1",
            "port": 5001,
            "debug": False,
            "default_speaker": "default_speaker.wav",
            "default_language": "en",
            "sample_rate": 24000
        }
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(loaded_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Создан файл конфигурации с настройками по умолчанию: {config_file}")
        except Exception as e:
            logger.error(f"Ошибка создания файла конфигурации: {e}")
    
    return loaded_config

def validate_config(config: Dict[str, Any]) -> bool:
    """Валидация конфигурации"""
    # Проверяем обязательные поля
    required_fields = ["default_speaker"]
    for field in required_fields:
        if field not in config:
            logger.error(f"В конфигурации отсутствует обязательное поле: {field}")
            return False
    
    # Проверяем существование файла голоса по умолчанию
    default_speaker = config.get("default_speaker")
    if default_speaker and not os.path.exists(default_speaker):
        logger.warning(f"Файл голоса по умолчанию не найден: {default_speaker}")
        # Создаем заглушку или пытаемся найти в стандартных местах
        if default_speaker == "default_speaker.wav":
            # Проверяем наличие в текущей директории
            if not os.path.exists("default_speaker.wav"):
                logger.error("Файл default_speaker.wav не найден в текущей директории.")
                # Создаем минимальную заглушку (1 секунда тишины)
                try:
                    from scipy.io import wavfile
                    import numpy as np
                    silence = np.zeros(24000, dtype=np.float32)  # 1 секунда тишины при 24кГц
                    wavfile.write("default_speaker.wav", 24000, silence)
                    logger.info("Создан файл-заглушка default_speaker.wav")
                except:
                    logger.error("Не удалось создать файл-заглушку. Установите scipy или предоставьте файл голоса.")
                    return False
    
    return True

# ========== Инициализация ==========
def init_tts_model():
    """Инициализация модели TTS"""
    global tts_model, sample_rate
    
    try:
        logger.info("Загрузка модели TTS...")
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        # Получаем частоту дискретизации модели
        sample_rate = tts_model.synthesizer.output_sample_rate
        logger.info(f"Модель TTS загружена. Sample rate: {sample_rate}")
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели TTS: {e}")
        raise

def init_audio():
    """Инициализация аудиосистемы"""
    pygame.mixer.init(frequency=sample_rate)
    logger.info(f"Аудиосистема инициализирована (freq={sample_rate})")

# ========== Функции TTS ==========
def generate_speech(text: str, speaker_wav: Optional[str] = None, language: Optional[str] = None) -> Optional[np.ndarray]:
    """Генерация речи из текста"""
    global tts_model, DEFAULT_SPEAKER, config
    
    if tts_model is None:
        logger.error("Модель TTS не инициализирована")
        return None
    
    # Используем голос по умолчанию, если не указан
    if speaker_wav is None:
        speaker_wav = DEFAULT_SPEAKER
    
    # Используем язык по умолчанию из конфига, если не указан
    if language is None:
        language = config.get('default_language', 'en')
    
    # Проверяем существование файла голоса
    if not os.path.exists(speaker_wav):
        logger.error(f"Файл голоса не найден: {speaker_wav}")
        # Пробуем использовать голос по умолчанию из конфига
        if speaker_wav != DEFAULT_SPEAKER:
            logger.info(f"Пробуем использовать голос по умолчанию: {DEFAULT_SPEAKER}")
            if os.path.exists(DEFAULT_SPEAKER):
                speaker_wav = DEFAULT_SPEAKER
            else:
                return None
    
    try:
        logger.info(f"Генерация речи: '{text[:50]}...' (голос: {os.path.basename(speaker_wav)}, язык: {language})")
        
        audio_data = tts_model.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            split_sentences=True
        )
        
        return np.array(audio_data)
        
    except Exception as e:
        logger.error(f"Ошибка генерации речи: {e}")
        return None

def save_audio_to_temp(audio_data: np.ndarray) -> str:
    """Сохранение аудиоданных во временный файл"""
    try:
        # Создаем временный файл
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        # Используем scipy для сохранения WAV, если установлен
        try:
            from scipy.io import wavfile
            wavfile.write(temp_file.name, sample_rate, audio_data.astype(np.float32))
        except ImportError:
            # Альтернатива: использовать soundfile
            try:
                import soundfile as sf
                sf.write(temp_file.name, audio_data, sample_rate)
            except ImportError:
                logger.warning("Не установлены scipy или soundfile. Используем raw сохранение.")
                # Простое сохранение raw данных (нестандартный формат)
                audio_data.astype(np.float32).tofile(temp_file.name)
        
        logger.info(f"Аудио сохранено во временный файл: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Ошибка сохранения аудио: {e}")
        return ""

# ========== Функции воспроизведения ==========
def audio_player_worker(audio_data: np.ndarray, stop_event: threading.Event):
    """Функция для воспроизведения аудио в отдельном потоке"""
    global is_playing, current_audio_data
    
    try:
        logger.info("Начинаю воспроизведение сгенерированной речи...")
        
        # Воспроизводим с помощью sounddevice
        sd.play(audio_data, sample_rate)
        
        # Ждем окончания воспроизведения или сигнала остановки
        while sd.get_stream().active and not stop_event.is_set():
            time.sleep(0.1)
        
        if stop_event.is_set():
            sd.stop()
            logger.info("Воспроизведение остановлено")
        else:
            logger.info("Воспроизведение завершено")
        
        sd.stop()
        is_playing = False
        current_audio_data = None
        stop_event.clear()
        
    except Exception as e:
        logger.error(f"Ошибка воспроизведения: {e}")
        is_playing = False
        current_audio_data = None
        stop_event.clear()

def play_audio_file(file_path: str, stop_event: threading.Event):
    """Воспроизведение аудиофайла через pygame"""
    global is_playing
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return
        
        logger.info(f"Воспроизведение файла: {file_path}")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            time.sleep(0.1)
            
        if stop_event.is_set():
            pygame.mixer.music.stop()
            logger.info("Воспроизведение остановлено")
        
        pygame.mixer.music.unload()
        is_playing = False
        stop_event.clear()
        
    except Exception as e:
        logger.error(f"Ошибка воспроизведения файла: {e}")
        is_playing = False
        stop_event.clear()

# ========== API Endpoints ==========
@app.route('/tts/speak', methods=['POST'])
def speak_text():
    """API endpoint для генерации и воспроизведения речи из текста"""
    global current_audio_data, is_playing, player_thread, stop_event, DEFAULT_SPEAKER, config
    
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type должен быть application/json"
        }), 400
    
    data = request.get_json()
    text = data.get('text')
    
    # Используем голос из запроса или по умолчанию
    speaker_wav = data.get('speaker_wav', DEFAULT_SPEAKER)
    
    # Используем язык из запроса или из конфига
    language = data.get('language', config.get('default_language', 'en'))
    
    if not text:
        return jsonify({
            "status": "error",
            "message": "Не указан text в теле запроса"
        }), 400
    
    # Проверка существования файла голоса
    if not os.path.exists(speaker_wav):
        logger.warning(f"Указанный файл голоса не найден: {speaker_wav}. Используется голос по умолчанию.")
        speaker_wav = DEFAULT_SPEAKER
        if not os.path.exists(speaker_wav):
            return jsonify({
                "status": "error",
                "message": f"Файл голоса не найден: {speaker_wav}. Проверьте конфигурацию."
            }), 404
    
    # Остановка текущего воспроизведения если играет
    if is_playing:
        stop_event.set()
        if player_thread:
            player_thread.join(timeout=2)
    
    # Генерация речи
    audio_data = generate_speech(text, speaker_wav, language)
    
    if audio_data is None:
        return jsonify({
            "status": "error",
            "message": "Ошибка генерации речи"
        }), 500
    
    # Запуск воспроизведения
    try:
        current_audio_data = audio_data
        is_playing = True
        stop_event.clear()
        
        player_thread = threading.Thread(
            target=audio_player_worker,
            args=(audio_data, stop_event),
            daemon=True
        )
        player_thread.start()
        
        logger.info(f"Запущено воспроизведение речи: '{text[:50]}...'")
        return jsonify({
            "status": "success",
            "message": "Начато воспроизведение сгенерированной речи",
            "text_length": len(text),
            "audio_samples": len(audio_data),
            "speaker": os.path.basename(speaker_wav),
            "language": language
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка запуска воспроизведения: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка запуска воспроизведения: {str(e)}"
        }), 500

@app.route('/tts/generate', methods=['POST'])
def generate_audio():
    """API endpoint только для генерации аудио (без воспроизведения)"""
    global DEFAULT_SPEAKER, config
    
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type должен быть application/json"
        }), 400
    
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({
            "status": "error",
            "message": "Не указан text в теле запроса"
        }), 400
    
    # Используем голос из запроса или по умолчанию
    speaker_wav = data.get('speaker_wav', DEFAULT_SPEAKER)
    
    # Используем язык из запроса или из конфига
    language = data.get('language', config.get('default_language', 'en'))
    
    # Проверка существования файла голоса
    if not os.path.exists(speaker_wav):
        logger.warning(f"Указанный файл голоса не найден: {speaker_wav}. Используется голос по умолчанию.")
        speaker_wav = DEFAULT_SPEAKER
        if not os.path.exists(speaker_wav):
            return jsonify({
                "status": "error",
                "message": f"Файл голоса не найден: {speaker_wav}"
            }), 404
    
    # Генерация речи
    audio_data = generate_speech(text, speaker_wav, language)
    
    if audio_data is None:
        return jsonify({
            "status": "error",
            "message": "Ошибка генерации речи"
        }), 500
    
    # Сохранение во временный файл
    temp_file = save_audio_to_temp(audio_data)
    
    if not temp_file:
        return jsonify({
            "status": "error",
            "message": "Ошибка сохранения аудио"
        }), 500
    
    # Отправка файла
    try:
        response = send_file(
            temp_file,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_speech.wav'
        )
        
        # Удаляем файл после отправки
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(temp_file)
                logger.info(f"Временный файл удален: {temp_file}")
            except:
                pass
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка отправки файла: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка отправки файла: {str(e)}"
        }), 500

@app.route('/tts/stop', methods=['POST'])
def stop_audio():
    """API endpoint для остановки воспроизведения"""
    global is_playing, stop_event, player_thread
    
    if not is_playing:
        return jsonify({
            "status": "info",
            "message": "Воспроизведение не активно"
        }), 200
    
    try:
        stop_event.set()
        
        if player_thread:
            player_thread.join(timeout=2)
        
        logger.info("Воспроизведение остановлено по запросу")
        return jsonify({
            "status": "success",
            "message": "Воспроизведение остановлено"
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка при остановке: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при остановке: {str(e)}"
        }), 500

@app.route('/tts/status', methods=['GET'])
def get_status():
    """API endpoint для получения статуса TTS сервиса"""
    global tts_model, is_playing, current_audio_data, DEFAULT_SPEAKER, config
    
    status = {
        "status": "success",
        "tts_model_loaded": tts_model is not None,
        "is_playing": is_playing,
        "sample_rate": sample_rate,
        "audio_data_available": current_audio_data is not None,
        "default_speaker": DEFAULT_SPEAKER,
        "default_language": config.get('default_language', 'en'),
        "config": config
    }
    
    if current_audio_data is not None:
        status["audio_samples"] = len(current_audio_data)
        status["audio_duration"] = len(current_audio_data) / sample_rate
    
    return jsonify(status), 200

@app.route('/tts/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global tts_model, DEFAULT_SPEAKER
    
    if tts_model is None:
        return jsonify({
            "status": "degraded",
            "message": "TTS модель не загружена"
        }), 503
    
    if DEFAULT_SPEAKER is None or not os.path.exists(DEFAULT_SPEAKER):
        return jsonify({
            "status": "degraded",
            "message": f"Файл голоса по умолчанию не найден: {DEFAULT_SPEAKER}"
        }), 503
    
    return jsonify({
        "status": "healthy",
        "service": "tts_microservice",
        "model": "xtts_v2",
        "default_speaker": os.path.basename(DEFAULT_SPEAKER)
    }), 200

@app.route('/tts/config/reload', methods=['POST'])
def reload_config():
    """Перезагрузка конфигурации"""
    global config, DEFAULT_SPEAKER
    
    try:
        new_config = load_config()
        if validate_config(new_config):
            config.update(new_config)
            DEFAULT_SPEAKER = config.get('default_speaker')
            logger.info("Конфигурация перезагружена")
            return jsonify({
                "status": "success",
                "message": "Конфигурация перезагружена",
                "config": config
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Ошибка валидации конфигурации"
            }), 400
    except Exception as e:
        logger.error(f"Ошибка перезагрузки конфигурации: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка перезагрузки конфигурации: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "Endpoint не найден"
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "status": "error",
        "message": "Внутренняя ошибка сервера"
    }), 500

# ========== Основная функция ==========
def main():
    """Основная функция запуска сервиса"""
    global config, DEFAULT_SPEAKER
    
    try:
        # 1. Загрузка конфигурации
        config = load_config()
        
        # 2. Валидация и установка голоса по умолчанию
        if validate_config(config):
            DEFAULT_SPEAKER = config.get('default_speaker')
            logger.info(f"Установлен голос по умолчанию: {DEFAULT_SPEAKER}")
        else:
            logger.error("Конфигурация не прошла валидацию. Используются значения по умолчанию.")
            DEFAULT_SPEAKER = "default_speaker.wav"
            config['default_speaker'] = DEFAULT_SPEAKER
        
        # 3. Загрузка модели TTS
        init_tts_model()
        
        # 4. Инициализация аудио
        init_audio()
        
        # 5. Получение параметров из конфига
        host = config.get('host', '127.0.0.1')
        port = int(config.get('port', 5001))
        debug = config.get('debug', False)
        
        logger.info(f"Запуск TTS сервиса на {host}:{port}")
        logger.info(f"Debug режим: {debug}")
        logger.info(f"Голос по умолчанию: {DEFAULT_SPEAKER}")
        logger.info(f"Язык по умолчанию: {config.get('default_language', 'en')}")
        
        app.run(host=host, port=port, debug=debug, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Остановка сервиса...")
        if is_playing:
            stop_event.set()
            if player_thread:
                player_thread.join(timeout=2)
        sd.stop()
        pygame.mixer.quit()
        logger.info("Сервис остановлен")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sd.stop()
        pygame.mixer.quit()
        raise

if __name__ == '__main__':
    main()