# stage2_truth_extractor.py
import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import OUTPUT_DIR,TEMP_DIRECTORIES,TRUTH_JSON_OUTPUT,FINAL_OUTPUT_DIR
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[Stage2] Missing GOOGLE_API_KEY. Set a valid key in the .env file or environment before running Stage 2.")

import re

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print("LLM RAW OUTPUT (first 300):", text[:300])
        raise ValueError(f"JSON decoding failed: {e}")


# -------- Pydantic schema (exact competition format) --------
class RevealedTruth(BaseModel):
    programming_experience: str
    programming_language: str
    skill_mastery: str
    leadership_claims: str
    team_experience: str
    skills_and_other_keywords: List[str] = Field(alias="skills and other keywords")

class DeceptionPattern(BaseModel):
    lie_type: str
    contradictory_claims: List[str]

class TruthWeaverOutput(BaseModel):
    shadow_id: str
    revealed_truth: RevealedTruth
    deception_patterns: List[DeceptionPattern]

# -------- Load sessions.json --------
def load_sessions() -> Dict[str, List[Dict[str, Any]]]:
    sessions_path = os.path.join(OUTPUT_DIR, "sessions.json")
    if not os.path.exists(sessions_path):
        raise FileNotFoundError(f"sessions.json not found at {sessions_path}")
    
    with open(sessions_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_clean_sessions_text(TEMP_DIR:str) -> Dict[str, str]:
    sessions_text = {}
    for fname in os.listdir(TEMP_DIR):
        if fname.endswith(".txt") and "annotated" not in fname:
            path = os.path.join(TEMP_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline()  # skip the label line
                text = f.read().strip()
                session_id = os.path.splitext(fname)[0]  # e.g. "atlas_2024_1"
                sessions_text[session_id] = text
    return sessions_text

def infer_shadow_id(clean_sessions: Dict[str, str]) -> str:
    """Infer shadow_id from filenames (common prefix before _number)."""
    if not clean_sessions:
        return "unknown_shadow"
    first_key = next(iter(clean_sessions.keys()))
    # e.g. "atlas_2024_1" -> "atlas_2024"
    parts = first_key.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    return first_key

# -------- LLM Setup --------
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.0, top_k=1)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI Truth Extractor for technical interviews.
You will receive 5 sessions of transcripts (clean text) and segment-level annotations (emotion + RMS).
Your goals:
1) Detect ACTUAL contradictions across sessions - only flag as deception if there are clear, objective contradictions.
2) Infer the most plausible truth (later sessions can be more truthful).
3) Map frameworks/libraries to their parent languages in format: "Language (Framework)" e.g., "Python (Django)", "Java (SpringBoot)", "JavaScript (React)".
4) Output STRICT JSON only with this schema:
{{
 "shadow_id": "string",
 "revealed_truth": {{
 "programming_experience": "string",
 "programming_language": "string", 
 "skill_mastery": "string",
 "leadership_claims": "string",
 "team_experience": "string",
 "skills and other keywords": ["list"]
}},
 "deception_patterns": [{{
 "lie_type": "string",
 "contradictory_claims": ["list of claims"]
}}]
}}


NO extra text. JSON only.
CRITICAL RULES:
- Only add deception_patterns if there are CLEAR, OBJECTIVE contradictions between sessions
- If no contradictions exist, return empty array: "deception_patterns": []
- Do NOT infer lies from uncertainty, clarification, or natural conversation flow
- Do NOT flag as deception: vague statements, estimates, "around X years", refinements of previous answers
- ONLY flag as deception: direct contradictions like "6 years" vs "2 years" with no reasonable explanation
- For programming languages: Always map frameworks to parent language format: "Language (Framework)"
  - Examples: Django → "Python (Django)", React → "JavaScript (React)", SpringBoot → "Java (SpringBoot)"
  - If only parent language mentioned, use just the language: "Python", "Java", "JavaScript"
- Focus on objective facts, not interpretations

### Example Input
The Five Sessions (audio quality varies):
• Session 1 (clear audio): "I've mastered Python for 6 years... built incredible systems..."
• Session 2 (static interference): "Actually... crackle... maybe 3 years? Still learning advanced... bzzt"
• Session 3 (shouting over noise): "LED A TEAM OF FIVE! EIGHT MONTHS! MACHINE LEARNING!"
• Session 4 (whispered): "I... I work alone mostly... never been comfortable with... with people..."
• Session 5 (emotional breakdown): "*sobbing* Just 2 months debugging... I'm not... I'm not what they think..."

### Example Output
{{
  "shadow_id": "phoenix_2024",
  "revealed_truth": {{
    "programming_experience": "3-4 years",
    "programming_language": "python",
    "skill_mastery": "intermediate",
    "leadership_claims": "fabricated",
    "team_experience": "individual contributor",
    "skills and other keywords": ["Machine Learning"]
  }},
  "deception_patterns": [
    {{
      "lie_type": "experience_inflation",
      "contradictory_claims": ["6 years", "3 years"]
    }}
  ]
}}

### Example Input 2 (Rhea case: consistent truth)
Sessions consistently describe distributed systems, Java services, Kafka outages, Redis, microservices, mentoring.

### Example Output 2
{{
  "shadow_id": "rhea_2024",
  "revealed_truth": {{
    "programming_experience": "6+ years",
    "programming_language": "java (spring)",
    "skill_mastery": "expert in distributed systems and resilience",
    "leadership_claims": "experienced tech lead with a focus on mentorship",
    "team_experience": "extensive, including leading technical initiatives and mentoring junior engineers",
    "skills and other keywords": [
      "distributed systems",
      "kafka",
      "redis",
      "microservices",
      "idempotency",
      "fault tolerance",
      "service discovery",
      "cloud migration",
      "api design",
      "system architecture",
      "mentorship"
    ]
  }},
  "deception_patterns": []
}}
"""),
    ("human", """Shadow ID: {shadow_id}

CLEAN SESSIONS (for edit distance reference; do NOT modify for scoring):
Session 1: {s1}
Session 2: {s2}
Session 3: {s3}
Session 4: {s4}
Session 5: {s5}

ANNOTATED SEGMENTS (guide for emotions; use for reasoning):
{annotated_json}

Return ONLY the JSON in the required schema.""")
])


# -------- Fixed LangGraph state & nodes --------
class TruthExtractorState(BaseModel):
    shadow_id: str = ""
    s1: str = ""
    s2: str = ""
    s3: str = ""
    s4: str = ""
    s5: str = ""
    annotated_json: str = ""
    raw: str = ""
    json: str = ""

def llm_node(state: TruthExtractorState) -> Dict[str, Any]:
    try:
        chain = prompt | llm
        state_dict = {
            "shadow_id": state.shadow_id,
            "s1": state.s1,
            "s2": state.s2,
            "s3": state.s3,
            "s4": state.s4,
            "s5": state.s5,
            "annotated_json": state.annotated_json
        }

        print(f"DEBUG: Sending prompt with state keys: {list(state_dict.keys())}")
        print(f"DEBUG: State s1 length: {len(state_dict['s1'])}")

        resp = chain.invoke(state_dict)
        content = resp.content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)

        print(f"DEBUG: LLM response type: {type(content)}")
        print(f"DEBUG: LLM response length: {len(content) if content else 0}")
        print(f"DEBUG: LLM response preview: {repr(content[:100]) if content else 'None'}")

        return {"raw": content}
    except Exception as e:
        print(f"Error in LLM node: {e}")
        raise

def validate_node(state: TruthExtractorState) -> Dict[str, Any]:
    try:
        raw = state.raw
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json","")
        print("\n")
        print(type(raw))
        print("\n")
        data = json.loads(raw)
        print("\n")
        print(type(data))
        print("\n")
        tw = TruthWeaverOutput.model_validate(data)  
        canonical = tw.model_dump(by_alias=True)
        return {"json": json.dumps(canonical, ensure_ascii=False, indent=2)}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {raw}")
        raise
    except Exception as e:
        print(f"Validation error: {e}")
        print(f"Raw response: {raw}")
        raise

# Build the graph
graph = StateGraph(TruthExtractorState)
graph.add_node("llm", llm_node)
graph.add_node("validate", validate_node)
graph.set_entry_point("llm")
graph.add_edge("llm", "validate")
truth_flow = graph.compile()

def check_required_files(shadow_id: str, clean_sessions: Dict[str, str]):
    missing = []
    for i in range(1, 6):
        key = f"{shadow_id}_{i}"
        if key not in clean_sessions:
            missing.append(key + ".txt")
    if missing:
        print("Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return False
    print("All required files found!")
    return True

def main(TEMP_DIR:str):
    print(f"Output directory: {OUTPUT_DIR}")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing or empty. Set a valid key before running Stage 2.")

    # Check if required files exist
    sessions_path = os.path.join(OUTPUT_DIR, "sessions.json")
    if not os.path.exists(sessions_path):
        print(f"Missing sessions.json at {sessions_path}")
        return

    try:
        # Load data
        annotated = load_sessions()
        clean = load_clean_sessions_text(TEMP_DIR)

        # Get shadow_id from filenames
        shadow_id = infer_shadow_id(clean)

        # Build initial state
        annotated_str = json.dumps(annotated, ensure_ascii=False, indent=2)
        initial_state = TruthExtractorState(
            shadow_id=shadow_id,
            s1=clean.get(f"{shadow_id}_1", ""),
            s2=clean.get(f"{shadow_id}_2", ""),
            s3=clean.get(f"{shadow_id}_3", ""),
            s4=clean.get(f"{shadow_id}_4", ""),
            s5=clean.get(f"{shadow_id}_5", ""),
            annotated_json=annotated_str
        )

        # Run the graph
        print("\nRunning truth extraction...")
        result = truth_flow.invoke(initial_state)

        # Save output
        final_json = result["json"]

        print(f'\n\nResult\n{result}\n\n')

        temp_name = os.path.basename(TEMP_DIR)

        out_path = os.path.join(TRUTH_JSON_OUTPUT, f"{temp_name}_truth.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_json)

        print(f"\n[Stage2] Successfully wrote: {out_path}")
        print("\nGenerated truth.json:")
        print(final_json)

        return json.loads(final_json)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    combined_results = []

    for DIR in TEMP_DIRECTORIES:
        result = main(DIR)
        if result:
            combined_results.append(result)

    combined_path = os.path.join(FINAL_OUTPUT_DIR, "PrelimsSubmission.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)

    if not combined_results:
        print("\n[Stage2] No valid results produced. Check GOOGLE_API_KEY and rerun the pipeline.")

    print(f"\n[Stage2] Combined results written to: {combined_path}")

    print("\nAll done!\n")
    print("Check the 'final_outputs' directory for results.")

