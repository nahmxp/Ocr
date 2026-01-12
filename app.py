from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import cv2
import numpy as np
from typing import Optional, List
import easyocr
import os

app = FastAPI(
    title="OCR API - Image to Text Extraction",
    description="Extract text from images using Tesseract OCR or EasyOCR",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Initialize EasyOCR reader (lazy loading)
_easyocr_reader = None

def get_easyocr_reader():
    """Lazy load EasyOCR reader"""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


def validate_image_file(file: UploadFile):
    """Validate if the uploaded file is an allowed image type"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def preprocess_image(image: Image.Image, enhance: bool = True) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy with multiple strategies
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if enhance:
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        processed = Image.fromarray(binary)
        processed = processed.filter(ImageFilter.SHARPEN)
        processed = processed.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    else:
        processed = Image.fromarray(gray)
    
    return processed


def extract_text_easyocr(image_bytes: bytes) -> str:
    """Extract text using EasyOCR - try multiple strategies and pick best"""
    try:
        reader = get_easyocr_reader()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        all_results = []
        
        # Strategy 1: Original image with allowlist
        try:
            res1 = reader.readtext(img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/_ ')
            text1 = ' '.join(res1)
            if len(text1) > 5:
                all_results.append(text1)
        except:
            pass
        
        # Strategy 2: Grayscale upscaled 2x
        try:
            upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res2 = reader.readtext(upscaled, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/_ ')
            text2 = ' '.join(res2)
            if len(text2) > 5:
                all_results.append(text2)
        except:
            pass
        
        # Strategy 3: CLAHE enhanced
        try:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            upscaled_enh = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res3 = reader.readtext(upscaled_enh, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/_ ')
            text3 = ' '.join(res3)
            if len(text3) > 5:
                all_results.append(text3)
        except:
            pass
        
        # Strategy 4: Binary threshold
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            upscaled_bin = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res4 = reader.readtext(upscaled_bin, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/_ ')
            text4 = ' '.join(res4)
            if len(text4) > 5:
                all_results.append(text4)
        except:
            pass
        
        # Pick the result that looks most like the expected format
        if not all_results:
            return ""
        
        # Score results based on expected patterns
        def score_result(text):
            score = 0
            if 'TYPE' in text.upper(): score += 10
            if 'S/N' in text.upper() or 'SN' in text.upper(): score += 10
            if 'PC-' in text.upper() or 'PC_' in text.upper(): score += 8
            if any(char.isdigit() for char in text): score += 5
            if '-' in text: score += 3
            score += len(text)  # Longer is usually better
            return score
        
        best_text = max(all_results, key=score_result)
        return best_text
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EasyOCR error: {str(e)}")


def extract_text_easyocr_detailed(image_bytes: bytes) -> dict:
    """Extract text with detailed information - multiple strategies"""
    try:
        reader = get_easyocr_reader()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try the strategy that works best
        best_results = None
        best_text = ""
        
        # Strategy 1: Upscaled grayscale
        try:
            upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res1 = reader.readtext(upscaled, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/_ ')
            text1 = ' '.join([r[1] for r in res1])
            if len(text1) > len(best_text):
                best_text = text1
                best_results = res1
        except:
            pass
        
        # Strategy 2: CLAHE enhanced
        try:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            upscaled_enh = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res2 = reader.readtext(upscaled_enh, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/_ ')
            text2 = ' '.join([r[1] for r in res2])
            if len(text2) > len(best_text):
                best_text = text2
                best_results = res2
        except:
            pass
        
        if not best_results:
            return {"text": "", "details": []}
        
        # Format detailed results
        details = []
        for bbox, text, confidence in best_results:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            details.append({
                "text": text,
                "confidence": float(confidence * 100),
                "left": int(min(x_coords)),
                "top": int(min(y_coords)),
                "width": int(max(x_coords) - min(x_coords)),
                "height": int(max(y_coords) - min(y_coords))
            })
        
        return {"text": best_text, "details": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EasyOCR error: {str(e)}")


@app.get("/")
def home():
    """Welcome endpoint with API information"""
    return {
        "message": "OCR API - Image to Text Extraction",
        "docs": "/docs",
        "endpoints": {
            "/api/ocr": "POST - Upload image for text extraction",
            "/api/ocr/detailed": "POST - Get detailed OCR with confidence scores",
            "/api/languages": "GET - List available OCR languages"
        }
    }


@app.post("/api/ocr")
async def extract_text(
    file: UploadFile = File(..., description="Image file to extract text from"),
    lang: str = Form("eng", description="Language code (e.g., 'eng', 'spa', 'fra')"),
    engine: str = Form("easyocr", description="OCR engine: 'tesseract' or 'easyocr' (recommended)"),
    preprocess: bool = Form(True, description="Apply image preprocessing (only for Tesseract)")
):
    """
    Extract text from an uploaded image using OCR
    
    - **file**: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
    - **lang**: Optional language code (default: 'eng') - for Tesseract only
    - **engine**: OCR engine to use: 'easyocr' (recommended, free) or 'tesseract'
    - **preprocess**: Apply image enhancement (default: True) - for Tesseract only
    
    Returns extracted text from the image
    """
    try:
        validate_image_file(file)
        contents = await file.read()
        
        if engine.lower() == "easyocr":
            extracted_text = extract_text_easyocr(contents)
            return {
                "success": True,
                "text": extracted_text,
                "filename": file.filename,
                "engine": "easyocr",
                "language": "auto-detected"
            }
        
        image = Image.open(io.BytesIO(contents))
        
        if preprocess:
            image = preprocess_image(image)
        
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 11',
            r'--oem 3 --psm 7',
        ]
        
        results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
                if text:
                    results.append(text)
            except:
                continue
        
        extracted_text = max(results, key=len) if results else ""
        
        return {
            "success": True,
            "text": extracted_text,
            "filename": file.filename,
            "engine": "tesseract",
            "language": lang,
            "preprocessed": preprocess
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr/detailed")
async def extract_text_detailed(
    file: UploadFile = File(..., description="Image file to extract text from"),
    lang: str = Form("eng", description="Language code (e.g., 'eng', 'spa', 'fra')"),
    engine: str = Form("easyocr", description="OCR engine: 'tesseract' or 'easyocr' (recommended)"),
    preprocess: bool = Form(True, description="Apply image preprocessing (only for Tesseract)")
):
    """
    Extract text with detailed information including confidence scores and bounding boxes
    
    - **file**: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
    - **lang**: Optional language code (default: 'eng') - for Tesseract only
    - **engine**: OCR engine to use: 'easyocr' (recommended, free) or 'tesseract'
    - **preprocess**: Apply image enhancement (default: True) - for Tesseract only
    
    Returns detailed OCR data with word positions and confidence levels
    """
    try:
        validate_image_file(file)
        contents = await file.read()
        
        if engine.lower() == "easyocr":
            result = extract_text_easyocr_detailed(contents)
            return {
                "success": True,
                "text": result["text"],
                "details": result["details"],
                "filename": file.filename,
                "engine": "easyocr",
                "language": "auto-detected"
            }
        
        image = Image.open(io.BytesIO(contents))
        
        if preprocess:
            image = preprocess_image(image)
        
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 11',
            r'--oem 3 --psm 7',
        ]
        
        best_text = ""
        best_data = None
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
                data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
                if len(text) > len(best_text):
                    best_text = text
                    best_data = data
            except:
                continue
        
        extracted_text = best_text
        ocr_data = best_data if best_data else {'text': [], 'conf': [], 'left': [], 'top': [], 'width': [], 'height': []}
        
        words_data = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:
                words_data.append({
                    'text': ocr_data['text'][i],
                    'confidence': float(ocr_data['conf'][i]),
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i]
                })
        
        return {
            "success": True,
            "text": extracted_text,
            "details": words_data,
            "filename": file.filename,
            "engine": "tesseract",
            "language": lang,
            "preprocessed": preprocess
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/languages")
def get_languages():
    """Get list of available OCR languages installed on the system"""
    try:
        languages = pytesseract.get_languages()
        return {
            "success": True,
            "languages": languages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
