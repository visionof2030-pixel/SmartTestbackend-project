from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import google.generativeai as genai
import os
import json
import itertools

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = img.filter(ImageFilter.MedianFilter())
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    return img

def ocr_image(img: Image.Image) -> str:
    config = "--psm 6"
    return pytesseract.image_to_string(
        img,
        lang="ara+eng",
        config=config
    )

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 50:
                text += page_text + "\n"
            else:
                img = page.to_image(resolution=300).original
                img = preprocess_image(img)
                text += ocr_image(img) + "\n"
    return text.strip()

def extract_text_from_image(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    img = preprocess_image(img)
    return ocr_image(img).strip()

def build_prompt(text: str, lang: str, count: int) -> str:
    lang_instruction = (
        "اكتب الناتج النهائي باللغة العربية الفصحى."
        if lang == "ar"
        else "Write the final output in clear academic English."
    )
    return f"""
{lang_instruction}

أنشئ {count} سؤال اختيار من متعدد من النص التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح موسع للإجابة الصحيحة
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

النص:
{text}
"""

@app.post("/ask-file")
async def ask_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    num_questions: int = Form(10)
):
    try:
        content = await file.read()
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
            text = extract_text_from_image(content)

        if len(text) < 100:
            raise HTTPException(status_code=400, detail="Extracted text too short")

        model = get_model()
        prompt = build_prompt(text, language, num_questions)
        response = model.generate_content(prompt)

        raw = response.text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

        return