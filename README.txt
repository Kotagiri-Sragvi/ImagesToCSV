Directory structure:
ImagesToCSV/
│
├── main.py
├── test.py
├── input/
│   ├── ImagesInput
│   └── (Your images)
└──output/
    ├── CSVoutput
    └── output.csv <- available once the program is executed

You also need to install Tesseract OCR binary, which can be done:
- Windows:
	Open powershell as administrator and execute:
	winget install --id UB-Mannheim.TesseractOCR
	or download it from- https://github.com/tesseract-ocr/tesseract
- Linux:
	(Ubuntu/Debian) sudo apt-get install tesseract-ocr
- MacOS:
	(Homebrew) brew install tesseract
