import os
import pytesseract
from img2table.document import Image
from img2table.ocr import TesseractOCR

# Explicit fallback path for Windows if PATH lookup fails
tesseract_bin = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_bin):
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin

# Initialize OCR engine
try:
    ocr = TesseractOCR(n_threads=2, lang="eng")
    print("Tesseract OCR and img2table successfully initialized.")
except Exception as e:
    print(f"Initialization failed: {e}")