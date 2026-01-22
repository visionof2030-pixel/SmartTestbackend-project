from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, re, tempfile, itertools, json
import pdfplumber
import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
from docx import Document
import google.generativeai as genai
import tiktoken

MODEL_NAME = "gemini-2.5-flash-lite"
MAX_RETRY = 2

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 2048
        }
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\w\u0600-\u06FF\s.,:;!?()-]', '', text)
    return text.strip()

def split_text(text: str, max_tokens=900):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

def extract_text_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_pdf_fitz(path: str) -> str:
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def preprocess_image(img: Image.Image) -> Image.Image:
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(gray)

def ocr_image(img: Image.Image, lang: str) -> str:
    lang_code = "ara+eng" if lang == "ar" else "eng"
    return pytesseract.image_to_string(img, lang=lang_code)

def quality_check(text: str) -> bool:
    if len(text) < 400:
        return False
    bad_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
    return bad_ratio < 0.35

def safe_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def lang_instruction(lang: str):
    return "Write the final output in clear academic English." if lang == "en" else "اكتب الناتج النهائي باللغة العربية الفصحى."

def build_prompt(topic: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح موسع وعميق للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- مستوى تعليمي واضح
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

الموضوع:
{topic}
"""

def generate_batch(topic: str, batch_size: int, language: str):
    batch_size = min(max(batch_size, 5), 20)
    for attempt in range(MAX_RETRY + 1):
        try:
            model = get_model()
            prompt = build_prompt(topic, language, batch_size)
            response = model.generate_content(prompt)
            data = safe_json(response.text)
            if not data or "questions" not in data:
                raise ValueError("Invalid JSON")
            if len(data["questions"]) < batch_size:
                raise ValueError("Insufficient questions")
            return data["questions"][:batch_size]
        except Exception:
            if attempt == MAX_RETRY:
                raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/ask-file")
async def ask_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    num_questions: int = Form(10)
):
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    text = ""

    if suffix == ".pdf":
        text = extract_text_pdf(path)
        if not quality_check(text):
            text = extract_text_pdf_fitz(path)
    elif suffix == ".docx":
        text = extract_text_docx(path)
    elif suffix in [".jpg", ".jpeg", ".png"]:
        img = preprocess_image(Image.open(path))
        text = ocr_image(img, language)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file")

    if not quality_check(text):
        raise HTTPException(status_code=422, detail="Low quality text")

    text = clean_text(text)
    chunks = split_text(text)

    all_questions = []
    remaining = num_questions

    for chunk in chunks:
        if remaining <= 0:
            break
        batch = generate_batch(chunk, remaining, language)
        all_questions.extend(batch)
        remaining -= len(batch)

    return {"questions": all_questions}