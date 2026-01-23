import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash-lite"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def vision_to_text(image_bytes: bytes, lang: str) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    model = genai.GenerativeModel(MODEL)

    prompt = (
        "اقرأ هذه الصورة واستخرج النص التعليمي منها بوضوح وترتيب."
        if lang == "ar"
        else "Read this image and extract the educational content clearly and in order."
    )

    response = model.generate_content([prompt, image])

    text = response.text.strip()

    if not text or len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="تعذر استخراج نص واضح من الصورة"
        )

    return text

def safe_json(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

def generate_questions(text: str, lang: str, count: int):
    model = genai.GenerativeModel(MODEL)

    prompt = f"""
{"اكتب باللغة العربية الفصحى." if lang=="ar" else "Write in clear academic English."}

أنشئ {count} سؤال اختيار من متعدد من النص التالي.

قواعد صارمة:
- 4 خيارات فقط
- شرح موسع ودقيق للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- أعد JSON فقط دون أي شرح إضافي

الصيغة:
{{
 "questions":[
  {{
   "question":"",
   "options":["","","",""],
   "answer":0,
   "explanations":["","","",""]
  }}
 ]
}}

النص:
{text}
"""

    r = model.generate_content(prompt)
    return safe_json(r.text)

def generate_flashcards(text: str, lang: str, count: int):
    model = genai.GenerativeModel(MODEL)

    prompt = f"""
{"اكتب بالعربية." if lang=="ar" else "Write in English."}

أنشئ {count} بطاقات تعليمية.
كل بطاقة تحتوي:
- front: مصطلح أو مفهوم
- back: تعريف واضح وربط مفاهيمي

أعد JSON فقط:
{{
 "cards":[
  {{
   "front":"",
   "back":""
  }}
 ]
}}

النص:
{text}
"""

    r = model.generate_content(prompt)
    return safe_json(r.text)

def generate_summary(text: str, lang: str):
    model = genai.GenerativeModel(MODEL)

    prompt = f"""
{"اكتب بالعربية." if lang=="ar" else "Write in English."}

لخّص النص مع التركيز على:
- التعريفات الأساسية
- ربط المفاهيم
- صياغة تعليمية واضحة
"""

    r = model.generate_content(prompt)
    return {"summary": r.text.strip()}

@app.post("/quiz")
async def quiz(
    text: str = Form(...),
    language: str = Form("ar"),
    number_of_questions: int = Form(10),
):
    return generate_questions(text, language, number_of_questions)

@app.post("/flashcards")
async def flashcards(
    text: str = Form(...),
    language: str = Form("ar"),
    number_of_questions: int = Form(10),
):
    return generate_flashcards(text, language, number_of_questions)

@app.post("/summary")
async def summary(
    text: str = Form(...),
    language: str = Form("ar"),
):
    return generate_summary(text, language)

@app.post("/file/quiz")
async def quiz_from_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    number_of_questions: int = Form(10),
):
    image_bytes = await file.read()
    text = vision_to_text(image_bytes, language)
    return generate_questions(text, language, number_of_questions)

@app.post("/file/flashcards")
async def flashcards_from_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    number_of_questions: int = Form(10),
):
    image_bytes = await file.read()
    text = vision_to_text(image_bytes, language)
    return generate_flashcards(text, language, number_of_questions)

@app.post("/file/summary")
async def summary_from_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
):
    image_bytes = await file.read()
    text = vision_to_text(image_bytes, language)
    return generate_summary(text, language)