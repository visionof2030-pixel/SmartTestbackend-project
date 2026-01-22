# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os, json, itertools

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

def build_prompt(text: str, lang: str, count: int):
    lang_instruction = (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى."
    )
    return f"""
{lang_instruction}

أنشئ {count} سؤال اختيار من متعدد من النص التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح للإجابة الصحيحة
- أعد JSON فقط

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

النص:
{text}
"""

@app.post("/ask-file")
async def ask_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    num_questions: int = Form(10)
):
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except:
        raise HTTPException(status_code=400, detail="Invalid text")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short")

    model = get_model()
    prompt = build_prompt(text, language, num_questions)
    response = model.generate_content(prompt)

    raw = response.text
    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start == -1 or end == -1:
        raise HTTPException(status_code=500, detail="Invalid model response")

    return json.loads(raw[start:end])