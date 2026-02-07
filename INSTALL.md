## 📋 INSTALL — Setup Guide for va_player

(Auto-generated, not seriously edited or checked, may contain inaccuracies)

### 📌 Project Architecture Overview

The project implements a voice assistant for audio player control with two core components:

| Component | Purpose | Model / Library | Environment |
|-----------|---------|-----------------|-------------|
| **Gemma-3n** | Speech recognition, tool call generation (XML requests) | `unsloth/gemma-3n-E4B-it` (4-bit quantized) | `venv_gemma` |
| **XTTS v2** | Speech synthesis (Text-to-Speech) | Coqui TTS `tts_models/multilingual/multi-dataset/xtts_v2` | `venv_xtts` |
| **audio_service** | Music playback via REST API | Flask + pygame | Can run in any environment |

---

### ⚙️ System Requirements

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y git python3.11 python3.11-venv python3-pip \
    libportaudio2 exiftool ffmpeg libsndfile1 portaudio19-dev

# For ROCm (AMD GPU) — optional, required for Gemma acceleration
# Follow official guide: https://rocm.docs.amd.com/
```

> **Note:** The Gemma model requires ~6 GB VRAM with 4-bit quantization. For CPU-only operation, change `load_in_4bit = False` → `load_in_8bit = True` in the code (performance will decrease).

---

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/mml1975/va_player.git
cd va_player
```

---

### 🔌 Step 2: Setup XTTS Environment (`tts_speak`)

```bash
# Create virtual environment
python3.11 -m venv venv_xtts
source venv_xtts/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install TTS sounddevice pygame flask numpy wavio

# Verify installation
python -c "from TTS.api import TTS; print('XTTS ready')"
```

> **Important:** The XTTS v2 model (~2.3 GB) will be automatically downloaded on first launch of `tts_service.py`.

---

### 🧠 Step 3: Setup Gemma Environment (`pipeline1`)

```bash
# Create separate virtual environment
python3.11 -m venv venv_gemma
source venv_gemma/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate bitsandbytes sentencepiece
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install jupyterlab ipykernel silero-vad wavio sounddevice

# Register environment as Jupyter kernel
python -m ipykernel install --user --name=venv_gemma --display-name "Python 3.11 (Gemma)"
```

> **Note:** 4-bit quantization (`load_in_4bit=True`) requires `bitsandbytes`. On some systems, compilation from source may be necessary.

---

### 🎵 Step 4: Setup Audio Service (`playerservice`)

```bash
# Can run in either environment; venv_xtts recommended
source venv_xtts/bin/activate

# Install additional dependencies
pip install flask pygame pandas mutagen
```

---

### 📂 Step 5: Generate Music Database

```bash
# Ensure exiftool is installed (see Step 1)
cd musicbase

# Scan music collection (replace /path/to/music with your path)
./scanm2.sh /path/to/music
python3 tune_exif.py

# Result: musicbase.csv created in project root
```

---

### ▶️ Step 6: Launch Services (in separate terminals)

**Terminal 1 — Audio Service:**
```bash
source venv_xtts/bin/activate
cd playerservice
python audio_service.py  # Listens on port 5000 by default
```

**Terminal 2 — TTS Service:**
```bash
source venv_xtts/bin/activate
cd tts_speak
python tts_service.py    # Listens on port 5001 by default
```

**Terminal 3 — Voice Assistant (Jupyter):**
```bash
source venv_gemma/bin/activate
cd pipeline1
jupyter lab
```
Open `va_ver1.ipynb`, select kernel **"Python 3.11 (Gemma)"**, and execute all cells sequentially.

---

### ⚙️ Configuration (Optional)

**tts_speak/tts_config.json** — TTS service settings:
```json
{
  "host": "127.0.0.1",
  "port": 5001,
  "debug": false,
  "default_speaker": "ls_female1.wav",
  "default_language": "en",
  "sample_rate": 24000
}
```

**playerservice/config.json** — Audio service settings (auto-generated on first launch).

---

### 🔒 Security Recommendations

1. All services bind to `127.0.0.1` by default — do not expose ports 5000/5001 externally without authentication.
2. Gemma model loads locally via Hugging Face — verify trusted source (`unsloth/gemma-3n-E4B-it`).
3. For production environments, add API keys to request headers between services.

---

### 🚨 Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| `OSError: PortAudio library not found` | Install `libportaudio2` and `portaudio19-dev` (see Step 1) |
| `bitsandbytes` installation fails | Use `pip install bitsandbytes --prefer-binary` or build from source |
| XTTS slow on first launch | Model (~2.3 GB) downloads automatically — wait for completion |
| Gemma requires excessive memory | Ensure `load_in_4bit = True` in code; for <6 GB VRAM use CPU mode |

---

### 📚 Project Structure

```
va_player/
├── musicbase/          # Scripts for music database generation
├── playerservice/      # Playback microservice (port 5000)
├── tts_speak/          # TTS microservice based on XTTS v2 (port 5001)
├── pipeline1/          # Jupyter notebook with Gemma-3n (va_ver1.ipynb)
├── mic_capture/        # Microphone capture scripts (Silero VAD)
├── query_dataset/      # Datasets for training/testing
├── voice1example/      # Example audio queries
└── doc/                # Documentation (description_en.md)
```

---

✅ After completing all steps, the system is ready: speak into the microphone — the assistant will recognize commands via Gemma and respond with synthesized speech via XTTS.
