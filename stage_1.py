# stage1_transcribe_annotate.py
import os
import csv
import json
from typing import List
import numpy as np
import librosa

from config import AUDIO_FILES, OUTPUT_DIR, WHISPER_BACKEND, WHISPER_MODEL, SER_MODEL_ID, RMS_SHOUT, RMS_WHISPER, RMS_STATIC
from utils_audio import EmotionClassifier, analyze_features

def ensure_dir(d): os.makedirs(d, exist_ok=True)

# ------------- Whisper loaders -------------
def load_whisper_model():
    if WHISPER_BACKEND == "faster-whisper":
        from faster_whisper import WhisperModel
        return WhisperModel(WHISPER_MODEL, device="auto")
    else:
        import whisper
        return whisper.load_model(WHISPER_MODEL)

def transcribe(whisper_model, path: str):
    if WHISPER_BACKEND == "faster-whisper":
        segments, info = whisper_model.transcribe(path, language="en", vad_filter=True)
        segs = []
        for i, s in enumerate(segments):
            segs.append({"id": i, "start": s.start, "end": s.end, "text": s.text.strip()})
        text = " ".join([s["text"] for s in segs]).strip()
        return {"text": text, "segments": segs}
    else:
        result = whisper_model.transcribe(path, language="en", task="transcribe", verbose=False)
        return {"text": result.get("text","").strip(), "segments": result.get("segments", [])}

# ------------- Build -------------

def build():
    ensure_dir(OUTPUT_DIR)
    ser = EmotionClassifier(SER_MODEL_ID)
    all_segments = []

    whisper_model = load_whisper_model()
    for idx, audio_path in enumerate(AUDIO_FILES, start=1):
        print(f"[Stage1] Processing Session {idx}: {audio_path}")
        result = transcribe(whisper_model, audio_path)
        segments = result["segments"]
        audio, sr = librosa.load(audio_path, sr=16000)

        clean_session_lines: List[str] = []
        annotated_session_lines: List[str] = []

        rms_mean = 0.0
        for seg in segments:
            start = int(seg["start"] * sr); end = int(seg["end"] * sr)
            chunk = audio[start:end]
            rms, pitch = analyze_features(chunk, sr)
            rms_mean += rms
        rms_mean /= len(segments)
        for seg in segments:
            start = int(seg["start"] * sr); end = int(seg["end"] * sr)
            chunk = audio[start:end]
            rms, pitch = analyze_features(chunk, sr)
            emotion = ser.predict_label(chunk, sr)

            style = ""
            if emotion == "angry" or rms > RMS_SHOUT*rms_mean:
                style = "[shouting]"
            elif emotion in ["sad", "fearful"] and rms < RMS_WHISPER*rms_mean:
                style = "[whispered]"
            elif emotion == "sad":
                style = "[sobbing]"
            elif rms < RMS_STATIC*rms_mean:
                style = "[static interference]"

            clean_session_lines.append(seg["text"].strip())
            annotated_session_lines.append(f"{style} {seg['text'].strip()}".strip())
            
            filename = os.path.splitext(os.path.basename(audio_path))[0] ### changed here
            all_segments.append({
                "session": filename, "start": float(seg["start"]), "end": float(seg["end"]),
                "text": seg["text"].strip(), "emotion": emotion, "rms": float(rms), "pitch": float(pitch)
            })

        filename = os.path.splitext(os.path.basename(audio_path))[0]  # e.g. "atlas_2025_1"
        with open(os.path.join(OUTPUT_DIR, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{filename}\n\n")
            f.write(" ".join(clean_session_lines).strip() + "\n")

        with open(os.path.join(OUTPUT_DIR, f"{filename}_annotated.txt"), "w", encoding="utf-8") as f:
            f.write(f"{filename} (annotated)\n\n")
            for line in annotated_session_lines:
                f.write(line + "\n")

    # CSV export
    with open(os.path.join(OUTPUT_DIR, "session_segments.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session","start","end","text","emotion","rms","pitch"])
        writer.writeheader(); writer.writerows(all_segments)

    # sessions.json
    sessions = {}
    for s in all_segments:
        sessions.setdefault(str(s["session"]), []).append({
            "start": s["start"], "end": s["end"], "text": s["text"],
            "emotion": s["emotion"], "rms": s["rms"], "pitch": s["pitch"]
        })
    with open(os.path.join(OUTPUT_DIR, "sessions.json"), "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build()
    print("\n[Stage1] Done. See outputs/ for transcripts & segments.")
