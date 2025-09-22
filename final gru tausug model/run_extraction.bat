@echo off
echo Tausug Text Extraction Pipeline
echo ==============================
echo.

echo Installing dependencies...
pip install PyPDF2==3.0.1 pdfplumber==0.10.3
echo.

echo Running text extraction...
python pdf_extractor.py
echo.

echo Pipeline completed!
echo Check the 'extracted_text' folder for results.
echo.
pause
