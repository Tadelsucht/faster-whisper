pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m faster_whisper ".\sample_audio_recording.mp3" --language en