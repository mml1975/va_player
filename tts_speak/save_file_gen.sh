curl -X POST http://127.0.0.1:5001/tts/generate \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world!", "speaker_wav": "ls_female1.wav"}' \
     --output generated.wav