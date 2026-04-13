import os
import json
import re
import time
import asyncio
import threading
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# ── OpenGradient ─────────────────────────────────────────────────────────────
OG_OK = False
llm_client = None
og = None
WORKING_MODEL = None
_ready = False
_init_done = False
_init_lock = threading.Lock()

MODEL_PRIORITY = [
    "CLAUDE_HAIKU_4_5",
    "CLAUDE_SONNET_4_5",
    "CLAUDE_SONNET_4_6",
    "GPT_5_MINI",
]

# ── Event loop ─────────────────────────────────────────────────────────────
_loop = None

def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

def _ensure_loop():
    global _loop
    if _loop is None:
        t = threading.Thread(target=_start_loop, daemon=True)
        t.start()
        deadline = time.time() + 10
        while _loop is None and time.time() < deadline:
            time.sleep(0.05)

def _run(coro, timeout=120):
    _ensure_loop()
    if _loop is None:
        raise RuntimeError("Event loop not ready")
    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)
    return asyncio.run_coroutine_threadsafe(_with_timeout(), _loop).result(timeout=timeout + 5)

# ── OG init ────────────────────────────────────────────────────────────────
def _init_og():
    global OG_OK, llm_client, og, _ready, _init_done, WORKING_MODEL
    with _init_lock:
        if _init_done:
            return
        _init_done = True
    try:
        import opengradient as _og
        import ssl
        import urllib3
        og = _og
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        private_key = os.environ.get("OG_PRIVATE_KEY", "")
        if not private_key:
            raise ValueError("OG_PRIVATE_KEY not set")
        print(f"OG_PRIVATE_KEY found: {private_key[:6]}...")
        llm_client = og.LLM(private_key=private_key)
        try:
            # Используем min_allowance=0.1 как в работающем скрипте
            approval = llm_client.ensure_opg_approval(min_allowance=0.1)
            print(f"OPG approval: {approval}")
        except Exception as e:
            print(f"Approval warning (continuing): {e}")
        OG_OK = True
        print("OG connected — selecting model...")
        _pick_model()
    except Exception as e:
        import traceback
        print(f"OG init failed: {e}\n{traceback.format_exc()}")
    finally:
        _ready = True
        print(f"OG ready. OG_OK={OG_OK}, model={WORKING_MODEL}")

def _pick_model():
    global WORKING_MODEL
    if not OG_OK or llm_client is None:
        return
    for name in MODEL_PRIORITY:
        if not hasattr(og.TEE_LLM, name):
            continue
        model = getattr(og.TEE_LLM, name)
        try:
            print(f"  Trying {name}...")
            result = _run(llm_client.chat(
                model=model,
                messages=[{"role": "user", "content": "Say: OK"}],
                max_tokens=5,
                temperature=0.0,
            ), timeout=90)
            raw = _extract_raw(result)
            if raw and raw.strip():
                WORKING_MODEL = model
                print(f"✓ Model selected: {name}")
                return
        except Exception as e:
            print(f"  {name} failed: {e}")
    print("WARNING: No working model found")

def _ensure_og():
    if not _init_done:
        t = threading.Thread(target=_init_og, daemon=True)
        t.start()
        t.join(timeout=180)

# ── Helpers ────────────────────────────────────────────────────────────────
def _extract_raw(result):
    if not result:
        return ""
    for attr in ['chat_output', 'completion_output', 'content', 'text', 'output']:
        val = getattr(result, attr, None)
        if val:
            if isinstance(val, dict) and val.get('content'):
                return str(val['content'])
            if isinstance(val, str) and val.strip():
                return val
    for attr in dir(result):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(result, attr)
            if callable(val):
                continue
            if isinstance(val, str) and val.strip() and len(val) > 2:
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
    _ensure_og()
    if not OG_OK or llm_client is None:
        return {"error": "OpenGradient not available"}
    if WORKING_MODEL is None:
        _pick_model()
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
            ), timeout=120)
            raw = _extract_raw(result)
            if not raw.strip():
                last_error = "Empty response"
                time.sleep(2)
                continue
            parsed = parse_json(raw)
            if "error" in parsed:
                last_error = parsed["error"]
                time.sleep(1)
                continue
            tx = getattr(result, "transaction_hash", None) or getattr(result, "payment_hash", None)
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
                _pick_model()
                if WORKING_MODEL is None:
                    break
            else:
                time.sleep(2)
    return {"error": f"All attempts failed: {last_error}"}

# ── System prompt (CV analyzer) ────────────────────────────────────────────
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
@app.route("/")
def index():
    return jsonify({
        "service": "CV Analyzer",
        "status": "ok",
        "og": OG_OK,
        "endpoints": ["/health", "/analyze", "/ui"]
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "og": OG_OK,
        "ready": _ready,
        "model": str(WORKING_MODEL) if WORKING_MODEL else None,
    })

@app.route("/ui")
def ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "cv-analyzer.html")
    if os.path.exists(html_path):
        return send_from_directory(base_dir, "cv-analyzer.html")
    else:
        return jsonify({"error": "UI file not found"}), 404

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    cv_text = (data.get("cv_text") or "").strip()
    pdf_base64 = data.get("pdf_base64")
    target_role = (data.get("target_role") or "").strip()

    if not cv_text and not pdf_base64:
        return jsonify({"error": "cv_text or pdf_base64 is required"}), 400

    print(f"\nAnalyzing CV | target_role: '{target_role}'")

    # Формируем текстовое сообщение, так как мультимодальность не везде поддерживается
    user_content = []
    if cv_text:
        user_content.append(f"CV TEXT:\n\n{cv_text}")
    if pdf_base64:
        # Лучше не отправлять PDF как document, а извлечь текст заранее (но для простоты предупредим)
        user_content.append("\n[PDF content was provided but text extraction is recommended on client side]")
        print("Warning: PDF binary not directly supported by all models. Consider extracting text client-side.")
    role_note = f"\n\nTarget role: {target_role}" if target_role else ""
    user_content.append(f"Please analyze this CV and return the JSON.{role_note}")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_content)}
    ]

    result = call_llm(messages)
    return jsonify(result)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    _ensure_og()  # синхронная инициализация перед запуском
    print(f"CV Analyzer on :{port} | OG: {'live' if OG_OK else 'demo'}, model: {WORKING_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=False)
