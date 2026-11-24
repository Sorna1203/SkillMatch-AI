import pdfplumber
import docx
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)


def extract_text(file):
    if not file or not file.filename:
        return ""

    try:
        # PDF
        if file.filename.endswith(".pdf"):

            text = ""

            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

            return text


        # DOCX
        elif file.filename.endswith(".docx"):

            document = docx.Document(file)
            return " ".join([p.text for p in document.paragraphs])


        # Image
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image = Image.open(file)
            return pytesseract.image_to_string(image)


        # Unsupported
        return ""

    except Exception as e:
        logger.error(f"Error extracting text from {file.filename}: {str(e)}")
        return ""
