# TruthLens
 
**An AI-powered deception detection pipeline that transcribes, analyzes, and cross-examines candidate audio testimonies to surface contradictions and identify the most plausible truth.**
 
---
 
## Overview
 
Truth Lens takes raw audio testimonies from a candidate and reconstructs a reliable picture of what they actually did — separating fact from exaggeration or fabrication. It combines speech transcription, emotional and acoustic analysis, and LLM-driven contradiction detection into a single two-stage pipeline.
 
Given five audio sessions per candidate, the system:
 
1. Transcribes what was said
2. Flags *how* it was said (tone, stress, interference)
3. Cross-references claims across sessions to catch inconsistencies
4. Produces a structured, schema-validated verdict on what's most likely true
## Why This Approach
 
Deception in spoken testimony rarely shows up as a single lie — it shows up as **drift**: a claim that shifts slightly between tellings, a confident tone that doesn't match a shaky detail, a term used once and never again. Truth Weaver is built to catch that drift by treating each candidate's testimony as a whole rather than five isolated clips.
 
## How It Works
 
### Stage 1 — Transcription & Annotation
 
- Converts each audio file to text using Whisper (`faster-whisper` on CPU, standard Whisper on GPU)
- Runs a HuggingFace speech emotion recognition model to tag emotional tone (e.g. confident, hesitant, stressed)
- Computes RMS volume features to detect shouting, whispering, or static interference
- Produces both a clean transcript (for scoring) and an annotated transcript (tagged with acoustic/emotional context)
- Outputs structured session data to `sessions.json`
### Stage 2 — Truth Extraction
 
- Feeds all five annotated sessions per candidate into Gemini via a LangChain + LangGraph pipeline
- The model cross-references claims across sessions, flags contradictions, and infers the most plausible underlying truth
- Output is validated against a strict Pydantic schema before being accepted — malformed or incomplete responses are rejected and retried, not silently passed through
- Produces a final, competition-ready JSON verdict per candidate
## Example Output
 
```json
{
  "shadow_id": "shadow_candidate_1",
  "revealed_truth": {
    "programming_experience": "3-4 years",
    "programming_language": "python",
    "skill_mastery": "intermediate",
    "leadership_claims": "fabricated",
    "team_experience": "individual contributor",
    "skills and other keywords": ["Machine Learning"]
  },
  "deception_patterns": [
    {
      "lie_type": "experience_inflation",
      "contradictory_claims": ["6 years", "3 years"]
    }
  ]
}
```
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Transcription | Whisper / faster-whisper |
| Emotion & acoustic analysis | HuggingFace Speech Emotion Recognition, librosa |
| Orchestration | LangChain, LangGraph |
| Reasoning & extraction | Google Gemini |
| Schema validation | Pydantic |
| Language | Python 3.12 |
 
## Project Structure
 
```
TruthLens/
├── inputs/                    Raw audio files (candidate_1.mp3 ... candidate_5.mp3)
├── outputs/                   Stage 1 intermediate outputs (transcripts, annotations, sessions.json)
├── final_outputs/             Final competition-ready results
│   ├── PrelimsSubmission.json
│   └── transcript.txt
├── config.py                  Model selection, audio thresholds, paths
├── utils_audio.py             Emotion classification + acoustic feature extraction
├── stage_1.py                 Transcription & annotation pipeline
├── stage_2.py                 LangGraph truth-extraction pipeline
├── pipeline.sh                Full pipeline runner (macOS/Linux)
├── run_all.bat                Full pipeline runner (Windows)
└── requirements.txt
```
 
## Getting Started
 
### Prerequisites
 
- Python 3.10+ (3.12 recommended)
- `ffmpeg`
- A Google Gemini API key
### Installation
 
```bash
git clone https://github.com/ananya-kn/TruthLens.git
cd TruthLens
 
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
 
pip install -r requirements.txt
```
 
### Configuration
 
Create a `.env` file in the project root:
 
```
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
```
 
### Running the Pipeline
 
Place five audio files per candidate in `inputs/`, named `<id>_1.mp3` through `<id>_5.mp3`, then run:
 
```bash
chmod +x pipeline.sh
./pipeline.sh
```
 
Results will be written to `final_outputs/PrelimsSubmission.json`.
 
## Configuration Options
 
`config.py` exposes:
 
- `WHISPER_BACKEND` — `"openai-whisper"` or `"faster-whisper"`
- `SER_MODEL_ID` — HuggingFace emotion classification model
- `RMS_SHOUT`, `RMS_WHISPER`, `RMS_STATIC` — acoustic thresholds for tone tagging
## Troubleshooting
 
| Issue | Fix |
|---|---|
| Out of memory (≤8GB RAM) | Run via Google Colab with GPU enabled |
| `Permission denied` running `pipeline.sh` | `chmod +x pipeline.sh` |
| `ffmpeg` not found | Install via your platform's package manager |
| API key errors | Confirm `GOOGLE_API_KEY` is set in `.env` with no surrounding quotes |
| Audio not detected | Confirm filenames follow the `<id>_N.mp3` convention |
 
## License
 
MIT
 
---
