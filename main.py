# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import pdfplumber
import io, os, json, itertools

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "gemini-2.5-flash-lite"

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

def extract_text(file: UploadFile, raw: bytes) -> str:
    name = file.filename.lower()

    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    return raw.decode("utf-8", errors="ignore")

def build_prompt(text: str, lang: str, count: int):
    if lang == "en":
        lang_rule = "Write the final output in clear English."
    else:
        lang_rule = "اكتب الناتج النهائي باللغة العربية الفصحى."

    return f"""
{lang_rule}

المحتوى التالي قد يكون:
- كلمة واحدة
- عبارة قصيرة
- موضوع عام
- أو نصًا طويلًا

المطلوب:
أنشئ {count} سؤال اختيار من متعدد اعتمادًا على المحتوى فقط.
إذا كان المحتوى كلمة واحدة أو موضوعًا عامًا:
استنتج الأسئلة من المعرفة العامة المرتبطة به.

قواعد صارمة:
- 4 خيارات لكل سؤال
- إجابة واحدة صحيحة
- شرح مختصر للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- أعد JSON فقط بدون أي نص إضافي

الصيغة:
{{
 "questions":[
  {{
   "q":"",
   "options":["","","",""],
   "answer":0,
   "explanations":["","","",""]
  }}
 ]
}}

المحتوى:
{text}
"""

@app.post("/ask-file")
async def ask_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    num_questions: int = Form(10)
):
    raw = await file.read()
    text = extract_text(file, raw).strip()

    if not text:
        raise HTTPException(status_code=400, detail="المحتوى فارغ")

    model = get_model()
    response = model.generate_content(
        build_prompt(text, language, num_questions)
    )

    raw_out = response.text or ""
    start = raw_out.find("{")
    end = raw_out.rfind("}") + 1

    if start == -1 or end == -1:
        raise HTTPException(status_code=500, detail="رد غير صالح من النموذج")

    return json.loads(raw_out[start:end])