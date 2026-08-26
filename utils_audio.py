# utils_audio.py
import numpy as np
import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from typing import Tuple

class EmotionClassifier:
    def __init__(self, model_id: str):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.eval()

    def predict_label(self, audio: np.ndarray, sr: int) -> str:
        if audio.size == 0:
            return "neutral"
        inputs = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
        return self.model.config.id2label[pred_id]

def analyze_features(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    return rms, pitch_mean
