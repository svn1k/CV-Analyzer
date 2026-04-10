import os
import json
import re
import time
import asyncio
import threading
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from flask import send_from_directory

load_dotenv()
app = Flask(__name__)
CORS(app)

# ── OpenGradient ──────────────────────────────────────────────────────────────
OG_OK = False
llm_client = None
WORKING_MODEL = None

# Единый event loop в отдельном потоке
_loop = None
_loop_thread = None

def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

def _run(coro):
    """Запустить корутину в фоновом event loop и дождаться результата."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=120)

MODEL_PRIORITY = [
    "GEMINI_2_5_FLASH_LITE",
    "GEMINI_2_5_FLASH",
    "CLAUDE_HAIKU_4_5",
    "GPT_5_MINI",
    "CLAUDE_SONNET_4_5",
    "CLAUDE_SONNET_4_6",
    "GEMINI_2_5_PRO",
    "CLAUDE_OPUS_4_5",
    "GPT_5",
    "O4_MINI",
]

try:
    import opengradient as og
    import ssl
    import urllib3

    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Запускаем фоновый event loop
    _loop_thread = threading.Thread(target=_start_loop, daemon=True)
    _loop_thread.start()
    time.sleep(0.2)  # ждём пока loop запустится

    private_key = os.environ["OG_PRIVATE_KEY"]
    llm_client = og.LLM(private_key=private_key)

    # Approve токенов при старте
    try:
        approval = llm_client.ensure_opg_approval(opg_amount=0.1)
        print(f"OPG approval: {approval}")
    except Exception as e:
        print(f"Approval warning: {e}")

    OG_OK = True
    print("OG connected")
except Exception as e:
    print(f"Demo mode: {e}")
    OG_OK = False
    llm_client = None


# ── Поиск рабочей модели ──────────────────────────────────────────────────────
def probe_models():
    global WORKING_MODEL
    if not OG_OK or llm_client is None:
        return

    print("Probing models...")
    for name in MODEL_PRIORITY:
        if not hasattr(og.TEE_LLM, name):
            continue
        model_enum = getattr(og.TEE_LLM, name)
        try:
            print(f"Testing {name}...")
            result = _run(llm_client.chat(
                model=model_enum,
                messages=[{"role": "user", "content": "Reply: OK"}],
                max_tokens=5,
                temperature=0.0,
            ))
            raw = extract_raw(result)
            if raw.strip():
                WORKING_MODEL = model_enum
                print(f"Using model: {name}")
                return
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    print("No working model found.")


# ── Извлечение текста из ответа ───────────────────────────────────────────────
def extract_raw(result):
    if not result:
        return ""
    co = getattr(result, 'chat_output', None)
    if co:
        if isinstance(co, dict):
            for k in ('content', 'text', 'message', 'response', 'output'):
                if co.get(k):
                    return str(co[k])
        elif isinstance(co, str):
            return co
    comp = getattr(result, 'completion_output', None)
    if comp and str(comp).strip():
        return str(comp)
    if hasattr(result, 'text'):
        return result.text
    for attr in dir(result):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(result, attr)
            if callable(val):
                continue
            if isinstance(val, str) and ('<JSON>' in val or '"overall_score"' in val):
                return val
        except:
            pass
    return ""


def parse_json(raw):
    if not raw or not raw.strip():
        return {"error": "Empty response"}
    m = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception as e:
            print(f"JSON parse error: {e}")
    m = re.search(r'\{[\s\S]*?"overall_score"[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {"error": "Parse failed", "raw": raw[:300]}


def call_llm(messages, retries=3):
    global WORKING_MODEL
    if not OG_OK or llm_client is None:
        return {"error": "OpenGradient not available"}

    if WORKING_MODEL is None:
        probe_models()
    if WORKING_MODEL is None:
        return {"error": "No working model found"}

    last_error = ""
    for attempt in range(retries):
        try:
            print(f"LLM attempt {attempt+1} | model: {WORKING_MODEL}")
            result = _run(llm_client.chat(
                model=WORKING_MODEL,
                messages=messages,
                max_tokens=3000,
                temperature=0.3,
            ))
            raw = extract_raw(result)
            if not raw.strip():
                last_error = "Empty response"
                time.sleep(2)
                continue
            parsed = parse_json(raw)
            if "error" in parsed:
                last_error = parsed["error"]
                time.sleep(1)
                continue
            tx = getattr(result, "transaction_hash", None)
            if tx:
                parsed["proof"] = {
                    "transaction_hash": tx,
                    "explorer_url": f"https://explorer.opengradient.ai/tx/{tx}",
                }
            return parsed
        except Exception as e:
            last_error = str(e)
            print(f"LLM error attempt {attempt+1}: {e}")
            if "402" in str(e):
                WORKING_MODEL = None
                probe_models()
                if WORKING_MODEL is None:
                    break
            else:
                time.sleep(2)

    return {"error": f"All attempts failed: {last_error}"}


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert HR consultant and career coach. Analyze the provided CV/resume and reply ONLY with valid JSON inside <JSON>...</JSON> tags. No text outside.

Return this exact structure:
<JSON>
{
  "candidate_name": "Full Name or 'Candidate'",
  "overall_score": 72,
  "summary": "2-3 sentence overall assessment with key observations.",
  "strengths": [
    "Clear quantified achievements in sales (increased revenue by 40%)",
    "Strong technical stack relevant to modern roles"
  ],
  "weaknesses": [
    "No cover letter or professional summary section",
    "Employment gaps not explained (2019-2021)"
  ],
  "improvements": [
    "Add a 3-5 sentence professional summary at the top",
    "Quantify achievements with numbers wherever possible",
    "Add links to GitHub/portfolio/LinkedIn"
  ],
  "skill_scores": [
    {"skill": "Technical Skills", "score": 80},
    {"skill": "Communication", "score": 60},
    {"skill": "Leadership", "score": 45},
    {"skill": "Industry Experience", "score": 70},
    {"skill": "Presentation", "score": 55}
  ],
  "job_matches": [
    {"title": "Senior Software Engineer", "match": "high"},
    {"title": "Tech Lead", "match": "mid"},
    {"title": "Backend Developer", "match": "high"},
    {"title": "Solutions Architect", "match": "mid"},
    {"title": "CTO", "match": "low"}
  ]
}
</JSON>

Rules:
- overall_score: integer 0-100. Be strict and honest:
  - 0-40: poor CV (vague descriptions, no achievements, missing sections)
  - 41-60: average CV (some substance but lacks quantification and polish)
  - 61-75: good CV (clear structure, some achievements, minor gaps)
  - 76-90: strong CV (quantified achievements, complete, well-structured)
  - 91-100: exceptional CV (rare, near-perfect)
  - A CV with NO quantified achievements should never score above 55
  - A CV with vague descriptions like "worked on", "helped with" should lose 10-15 points
- strengths: 3-5 specific positive observations with evidence from the CV
- weaknesses: 3-5 specific issues found in the CV
- improvements: 4-6 actionable, concrete recommendations
- skill_scores: assess 5 dimensions, score 0-100 each
- job_matches: 5-8 relevant job titles with match level (high/mid/low)
- If target_role is provided, prioritize analysis toward that role
- Be honest and constructive, not generic
"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "og": OG_OK,
        "model": str(WORKING_MODEL) if WORKING_MODEL else None,
    })
@app.route("/ui")
def ui():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "cv-analyzer.html")
@app.route("/")
def index():
    return jsonify({
        "service": "CV Analyzer",
        "status": "ok",
        "og": OG_OK,
        "endpoints": ["/health", "/analyze"]
    })
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    cv_text = (data.get("cv_text") or "").strip()
    pdf_base64 = data.get("pdf_base64")
    target_role = (data.get("target_role") or "").strip()

    if not cv_text and not pdf_base64:
        return jsonify({"error": "cv_text or pdf_base64 is required"}), 400

    print(f"\nAnalyzing CV | target_role: '{target_role}'")

    user_content = []

    if pdf_base64:
        try:
            user_content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_base64
                }
            })
            print("PDF attached")
        except Exception as e:
            print(f"PDF error: {e}")
            return jsonify({"error": "Failed to process PDF"}), 400

    if cv_text:
        user_content.append({"type": "text", "text": f"CV TEXT:\n\n{cv_text}"})

    role_note = f"\n\nTarget role: {target_role}" if target_role else ""
    user_content.append({"type": "text", "text": f"Please analyze this CV and return the JSON.{role_note}"})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]["text"]}
    ]

    result = call_llm(messages)
    return jsonify(result)


# СТАЛО:
# Probe models at startup (before first request)
if OG_OK:
    threading.Thread(target=probe_models, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"CV Analyzer on :{port} | OG: {'live' if OG_OK else 'demo'}")
    app.run(host="0.0.0.0", port=port, debug=False)