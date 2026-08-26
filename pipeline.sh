#!/bin/bash
set -e  

python3 create_env.py

echo "[Pipeline] Stage 1: Transcribing & annotating..."
python3 stage_1.py

python3 merge_sessions.py
python3 create_folders.py

echo "[Pipeline] Stage 2: Truth extraction with Gemini..."
python3 stage_2.py
