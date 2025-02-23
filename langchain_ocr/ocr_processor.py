from typing import Dict, Any
from PIL import Image
import pytesseract
from text_processor import TextProcessor
from type import AgentState


class OCRProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()

    def process_image(self, state: AgentState) -> AgentState:
        try:
            # Use Tesseract OCR
            image = Image.open(state['image_path'])
            config = (
                '--oem 3 '  # Use LSTM OCR Engine Mode
                '--psm 1 --psm 3 --psm 4 '
                '-l eng '   # English language
                '--dpi 500 '  # Set DPI for better accuracy
                '-c preserve_interword_spaces=1 '  # Preserve word spacing
                '-c tessedit_do_invert=0 '  # Don't invert colors
                '-c textord_heavy_nr=1 '  # Reduce noise removal aggressiveness
                'pdftotext '
                '-c tessedit_create_txt=1 '  # Create text output
            )
            text = pytesseract.image_to_string(image, config=config)

            state['raw_text'] = text
            state['cleaned_text'] = self.text_processor.clean_text(text)
            return state
        except Exception as e:
            state['error'] = f"OCR processing failed: {str(e)}"
            return state
