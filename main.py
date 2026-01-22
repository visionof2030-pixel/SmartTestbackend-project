from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import re
import tempfile
import pdfplumber
import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
from docx import Document
from langdetect import detect
import google.generativeai as genai
import tiktoken

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = genai.GenerativeModel("gemini-pro")

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\w\u0600-\u06FF\s.,:;!?()-]', '', text)
    return text.strip()

def split_text(text: str, max_tokens=900):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = enc.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def extract_text_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        pass
    return text.strip()

def extract_text_pdf_fitz(file_path: str) -> str:
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

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

def generate_ai(prompt: str) -> str:
    response = MODEL.generate_content(prompt)
    return response.text

@app.post("/ask-file")
async def ask_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    num_questions: int = Form(10),
    mode: str = Form("questions")
):
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    extracted_text = ""

    if suffix in [".pdf"]:
        extracted_text = extract_text_pdf(tmp_path)
        if not quality_check(extracted_text):
            extracted_text = extract_text_pdf_fitz(tmp_path)

    elif suffix in [".docx"]:
        extracted_text = extract_text_docx(tmp_path)

    elif suffix in [".jpg", ".jpeg", ".png"]:
        img = Image.open(tmp_path)
        img = preprocess_image(img)
        extracted_text = ocr_image(img, language)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if not quality_check(extracted_text):
        raise HTTPException(
            status_code=422,
            detail="Low quality text extracted"
        )

    extracted_text = clean_text(extracted_text)
    chunks = split_text(extracted_text)

    final_prompt = ""

    if mode == "questions":
        final_prompt = f"""
اعتمد فقط على النص التالي وأنشئ {num_questions} أسئلة اختيار من متعدد مع الإجابات والتفسير بصيغة JSON:
{text}
"""
    elif mode == "flashcards":
        final_prompt = f"""
اعتمد فقط على النص التالي وأنشئ {num_questions} بطاقات تعليمية بصيغة JSON:
{text}
"""
    elif mode == "summary":
        final_prompt = f"""
لخّص النص التالي تلخيصًا واضحًا ومباشرًا:
{text}
"""
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    results = []
    for chunk in chunks:
        prompt = final_prompt.replace("{text}", chunk)
        results.append(generate_ai(prompt))

    return {
        "result": "\n".join(results)
    }
