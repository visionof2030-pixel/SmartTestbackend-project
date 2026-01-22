# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
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
key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

def extract_text_from_image(raw: bytes, lang: str) -> str:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_lang = "ara" if lang == "ar" else "eng"
    text = pytesseract.image_to_string(thresh, lang=ocr_lang, config="--psm 6")
    return text.strip()

def extract_text(file: UploadFile, raw: bytes, lang: str) -> str:
    name = file.filename.lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    if name.endswith((".png",".jpg",".jpeg")):
        return extract_text_from_image(raw, lang)
    return ""

def build_prompt(text: str, lang: str, count: int):
    rule = "Write in clear English." if lang=="en" else "اكتب الناتج النهائي باللغة العربية الفصحى."
    return f"""
{rule}

المحتوى قد يكون كلمة واحدة أو موضوعًا عامًا أو نصًا طويلًا.

المطلوب:
- أنشئ {count} سؤال اختيار من متعدد
- شرح موسع ودقيق للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
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
    text = extract_text(file, raw, language).strip()
    if not text:
        raise HTTPException(status_code=400, detail="تعذر قراءة المحتوى")
    model = get_model()
    response = model.generate_content(build_prompt(text, language, num_questions))
    raw_out = response.text or ""
    s = raw_out.find("{")
    e = raw_out.rfind("}") + 1
    if s==-1 or e==-1:
        raise HTTPException(status_code=500, detail="رد غير صالح")
    return json.loads(raw_out[s:e])