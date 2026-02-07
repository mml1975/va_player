#!/usr/bin/env python3
"""
Микросервис аудиоплеера для Linux
Требует установки: flask, pygame
"""

import os
import threading
import time
from flask import Flask, request, jsonify
import pygame
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Глобальные переменные для управления воспроизведением
current_file = None
is_playing = False
player_thread = None
stop_event = threading.Event()

def init_audio():
    """Инициализация аудиосистемы"""
    pygame.mixer.init()
    logger.info("Аудиосистема инициализирована")

def audio_player_worker(file_path, stop_event):
    """Функция для воспроизведения аудио в отдельном потоке"""
    global is_playing, current_file
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return
        
        logger.info(f"Начинаю воспроизведение: {file_path}")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            time.sleep(0.1)
            
        if stop_event.is_set():
            pygame.mixer.music.stop()
            logger.info("Воспроизведение остановлено")
        
        pygame.mixer.music.unload()
        is_playing = False
        current_file = None
        stop_event.clear()
        
    except Exception as e:
        logger.error(f"Ошибка воспроизведения: {e}")
        is_playing = False
        current_file = None
        stop_event.clear()

@app.route('/play', methods=['POST'])
def play_audio():
    """API endpoint для воспроизведения аудиофайла"""
    global current_file, is_playing, player_thread, stop_event
    
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type должен быть application/json"
        }), 400
    
    data = request.get_json()
    file_path = data.get('file_path')
    
    if not file_path:
        return jsonify({
            "status": "error",
            "message": "Не указан file_path в теле запроса"
        }), 400
    
    # Проверка существования файла
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": f"Файл не найден: {file_path}"
        }), 404
    
    # Проверка расширения файла
    allowed_extensions = ['.mp3', '.wav', '.ogg', '.flac']
    if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({
            "status": "error",
            "message": f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        }), 400
    
    # Остановка текущего воспроизведения если играет
    if is_playing:
        stop_event.set()
        if player_thread:
            player_thread.join(timeout=2)
    
    # Запуск нового воспроизведения
    try:
        current_file = file_path
        is_playing = True
        stop_event.clear()
        
        player_thread = threading.Thread(
            target=audio_player_worker,
            args=(file_path, stop_event),
            daemon=True
        )
        player_thread.start()
        
        logger.info(f"Запущено воспроизведение: {file_path}")
        return jsonify({
            "status": "success",
            "message": f"Начато воспроизведение: {os.path.basename(file_path)}",
            "file": file_path
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка запуска воспроизведения: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка запуска воспроизведения: {str(e)}"
        }), 500

@app.route('/stop', methods=['POST'])
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

@app.route('/status', methods=['GET'])
def get_status():
    """API endpoint для получения статуса плеера"""
    return jsonify({
        "status": "success",
        "is_playing": is_playing,
        "current_file": current_file,
        "volume": pygame.mixer.music.get_volume() if pygame.mixer.get_init() else 0
    }), 200

@app.route('/volume', methods=['POST'])
def set_volume():
    """API endpoint для установки громкости (0.0 - 1.0)"""
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type должен быть application/json"
        }), 400
    
    data = request.get_json()
    volume = data.get('volume')
    
    if volume is None:
        return jsonify({
            "status": "error",
            "message": "Не указан volume в теле запроса"
        }), 400
    
    try:
        volume = float(volume)
        if volume < 0 or volume > 1:
            return jsonify({
                "status": "error",
                "message": "Громкость должна быть в диапазоне от 0.0 до 1.0"
            }), 400
        
        pygame.mixer.music.set_volume(volume)
        logger.info(f"Установлена громкость: {volume}")
        return jsonify({
            "status": "success",
            "message": f"Громкость установлена на {volume}"
        }), 200
        
    except ValueError:
        return jsonify({
            "status": "error",
            "message": "Громкость должна быть числом"
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "audio_player_microservice"
    }), 200

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

def main():
    """Основная функция запуска сервиса"""
    try:
        init_audio()
        
        # Получение параметров из переменных окружения
        host = os.getenv('AUDIO_SERVICE_HOST', '127.0.0.1')
        port = int(os.getenv('AUDIO_SERVICE_PORT', '5000'))
        debug = os.getenv('AUDIO_SERVICE_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Запуск аудиосервиса на {host}:{port}")
        app.run(host=host, port=port, debug=debug, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Остановка сервиса...")
        if is_playing:
            stop_event.set()
            if player_thread:
                player_thread.join(timeout=2)
        pygame.mixer.quit()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        pygame.mixer.quit()
        raise

if __name__ == '__main__':
    main()
