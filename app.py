from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io
from typing import Optional, List

app = FastAPI(
    title="OCR API - Image to Text Extraction",
    description="Extract text from images using Tesseract OCR",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


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
    lang: str = Form("eng", description="Language code (e.g., 'eng', 'spa', 'fra')")
):
    """
    Extract text from an uploaded image using OCR
    
    - **file**: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
    - **lang**: Optional language code (default: 'eng')
    
    Returns extracted text from the image
    """
    try:
        # Validate file
        validate_image_file(file)
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform OCR
        extracted_text = pytesseract.image_to_string(image, lang=lang)
        
        return {
            "success": True,
            "text": extracted_text,
            "filename": file.filename,
            "language": lang
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr/detailed")
async def extract_text_detailed(
    file: UploadFile = File(..., description="Image file to extract text from"),
    lang: str = Form("eng", description="Language code (e.g., 'eng', 'spa', 'fra')")
):
    """
    Extract text with detailed information including confidence scores and bounding boxes
    
    - **file**: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
    - **lang**: Optional language code (default: 'eng')
    
    Returns detailed OCR data with word positions and confidence levels
    """
    try:
        # Validate file
        validate_image_file(file)
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # Extract full text
        extracted_text = pytesseract.image_to_string(image, lang=lang)
        
        # Format detailed results
        words_data = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:  # Filter out low confidence
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
            "language": lang
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
