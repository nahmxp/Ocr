# OCR API - Image to Text Extraction

A FastAPI-based REST API that extracts text from images using Tesseract OCR.

## Features

- 🖼️ Extract text from images (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
- 🌍 Support for multiple languages
- 📊 Detailed OCR output with confidence scores and bounding boxes
- 🚀 Fast and modern REST API with FastAPI
- 📚 Automatic interactive API documentation (Swagger UI)
- ✅ Input validation and error handling

## Prerequisites

- Python 3.8+
- Tesseract OCR engine

### Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd /media/xpert-ai/Documents/NDEV/Ocr
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Start the API server:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### 📚 Interactive Documentation

FastAPI provides automatic interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly in your browser using the Swagger UI!

## API Endpoints

### 1. Home - `GET /`
Returns API information and available endpoints.

**Example:**
```bash
curl http://localhost:8000/
```

### 2. Extract Text (Basic) - `POST /api/ocr`
Upload an image and get extracted text.

**Parameters:**
- `file` (required): Image file
- `lang` (optional): Language code (default: 'eng')

**Example:**
```bash
curl -X POST -F "file=@image.png" http://localhost:8000/api/ocr
```

**Response:**
```json
{
  "success": true,
  "text": "Extracted text from the image...",
  "filename": "image.png",
  "language": "eng"
}
```

### 3. Extract Text (Detailed) - `POST /api/ocr/detailed`
Get text extraction with confidence scores and bounding boxes.

**Example:**
```bash
curl -X POST -F "file=@image.png" http://localhost:8000/api/ocr/detailed
```

**Response:**
```json
{
  "success": true,
  "text": "Full extracted text...",
  "details": [
    {
      "text": "word",
      "confidence": 95.5,
      "left": 100,
      "top": 50,
      "width": 80,
      "height": 30
    }
  ],
  "filename": "image.png",
  "language": "eng"
}
```

### 4. Get Available Languages - `GET /api/languages`
Returns list of installed OCR languages.

**Example:**
```bash
curl http://localhost:8000/api/languages
```

## Testing with Python

```python
import requests

# Basic text extraction
with open('image.png', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/ocr', files=files)
    print(response.json())

# With specific language
with open('image.png', 'rb') as f:
    files = {'file': f}
    data = {'lang': 'eng'}
    response = requests.post('http://localhost:8000/api/ocr', files=files, data=data)
    print(response.json())
```

## Testing with cURL

```bash
# Basic OCR
curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/api/ocr

# OCR with Spanish language
curl -X POST -F "file=@/path/to/image.png" -F "lang=spa" http://localhost:8000/api/ocr

# Detailed OCR with bounding boxes
curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/api/ocr/detailed

# Get available languages
curl http://localhost:8000/api/languages
```

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- TIFF
- WEBP

## Language Support

To use different languages, install additional Tesseract language packs:

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr-spa  # Spanish
sudo apt-get install tesseract-ocr-fra  # French
sudo apt-get install tesseract-ocr-deu  # German
```

Common language codes:
- `eng` - English
- `spa` - Spanish
- `fra` - French
- `deu` - German
- `ara` - Arabic
- `chi_sim` - Chinese Simplified
- `jpn` - Japanese

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (missing file, invalid format)
- `500` - Server error (OCR processing failed)

## Configuration

Edit [app.py](app.py) to modify:
- `ALLOWED_EXTENSIONS`: Supported file types
- Port and host settings in `uvicorn.run()`
- API title and description in `FastAPI()` initialization

## Project Structure

```
Ocr/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── .gitignore         # Git ignore file
```

## Troubleshooting

**Issue: "pytesseract.pytesseract.TesseractNotFoundError"**
- Solution: Install Tesseract OCR engine and ensure it's in your PATH

**Issue: Low OCR accuracy**
- Solution: Use higher quality images, ensure good contrast, and select the correct language

**Issue: Port already in use**
- Solution: Change the port in `uvicorn.run()` at the bottom of [app.py](app.py)

## License

NDEV License - Feel free to use and modify for your projects.
