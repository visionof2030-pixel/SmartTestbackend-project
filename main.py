import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_VISION = "gemini-2.5-flash-lite"
MODEL_TEXT = "gemini-2.5-flash-lite"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def vision_to_text(image_bytes: bytes, lang: str):
    model = genai.GenerativeModel(MODEL_VISION)
    prompt = (
        "اقرأ هذه الصورة واستخرج محتواها التعليمي بدقة وبترتيب واضح."
        if lang == "ar"
        else "Read this image and extract the educational content clearly and accurately."
    )
    image_b64 = base64.b64encode(image_bytes).decode()
    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": image_b64},
            prompt,
        ]
    )
    return response.text.strip()

def generate_questions(text: str, lang: str, count: int):
    model = genai.GenerativeModel(MODEL_TEXT)
    prompt = f"""
{"اكتب باللغة العربية الفصحى." if lang=="ar" else "Write in clear academic English."}

أنشئ {count} سؤال اختيار من متعدد من النص التالي.

الشروط:
- 4 خيارات
- شرح موسع للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- أعد JSON فقط

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
    data = json.loads(r.text[r.text.find("{"):r.text.rfind("}")+1])
    return data

def generate_cards(text: str, lang: str, count: int):
    model = genai.GenerativeModel(MODEL_TEXT)
    prompt = f"""
{"اكتب بالعربية." if lang=="ar" else "Write in English."}

أنشئ {count} بطاقات تعليمية (تعريف ← شرح).

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
    return json.loads(r.text[r.text.find("{"):r.text.rfind("}")+1])

def generate_summary(text: str, lang: str):
    model = genai.GenerativeModel(MODEL_TEXT)
    prompt = f"""
{"اكتب بالعربية." if lang=="ar" else "Write in English."}

لخّص النص مع:
- تعريفات أساسية
- ربط مفاهيم
- نقاط واضحة

النص:
{text}
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
    return generate_cards(text, language, number_of_questions)

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
async def cards_from_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    number_of_questions: int = Form(10),
):
    image_bytes = await file.read()
    text = vision_to_text(image_bytes, language)
    return generate_cards(text, language, number_of_questions)

@app.post("/file/summary")
async def summary_from_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
):
    image_bytes = await file.read()
    text = vision_to_text(image_bytes, language)
    return generate_summary(text, language)