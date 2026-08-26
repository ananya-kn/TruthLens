# config.py
import os
import glob
import torch

if torch.cuda.is_available():
    # Use the standard OpenAI backend when a CUDA-enabled GPU is found
    WHISPER_BACKEND = "openai-whisper"
    print("✅ GPU detected. Using 'openai-whisper' backend.")
else:
    # Fall back to faster-whisper for more efficient CPU processing
    WHISPER_BACKEND = "faster-whisper"
    print("⚠️ No GPU detected. Falling back to 'faster-whisper' for CPU.")

WHISPER_MODEL = "large-v3"

AUDIO_DIR = "audio"
AUDIO_FILES = [f.replace("\\", "/") for f in glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))]

OUTPUT_DIR = "outputs"
FINAL_OUTPUT_DIR = "final_outputs"
TRUTH_JSON_OUTPUT = "truth_json_output"
SER_MODEL_ID = "superb/hubert-large-superb-er"
TEMP_DIRECTORIES = ["atlas", "oceanus", "rhea", "selene", "titan", "hyperion", "eos"]

# Heuristic thresholds
RMS_WHISPER = 0.5
RMS_STATIC = 0.2
RMS_SHOUT = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRUTH_JSON_OUTPUT, exist_ok=True)
