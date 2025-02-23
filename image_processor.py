import os
import json
import requests
from typing import Dict, Any
import pytesseract
import cv2
import numpy as np
import logging
import time
from text_processor import TextProcessor
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize text processor
text_processor = TextProcessor()

# LLM Configuration
MODEL_NAME = "fused_3b"


def process_with_llm(text: str) -> Dict[str, Any]:
    """Process text with local LLM to extract structured receipt data.

    Args:
        text (str): Cleaned OCR text

    Returns:
        Dict[str, Any]: Structured receipt data in JSON format
    """
    try:
        # Define the receipt schema
        receipt_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "receipt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "store_name": {"type": "string"},
                        "store_phone": {"type": "string"},
                        "date": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "number": {"type": "integer"},
                                    "price_single": {"type": "number"},
                                    "price_total": {"type": "number"},
                                    "vat_code": {"type": "string"}
                                },
                                "required": ["name", "number", "price_single", "price_total", "vat_code"]
                            }
                        },
                        "sub_total": {"type": "number"},
                        "tax": {"type": "number"},
                        "tip": {"type": "number"},
                        "total": {"type": "number"}
                    },
                    "required": ["store_name", "date", "items", "sub_total", "tax", "tip", "total"]
                }
            }
        }

        # Initialize OpenAI client for local LM Studio
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )

        # Create chat completion with schema validation
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an advanced receipt processing AI. Extract structured data from receipt text and format it according to the specified JSON schema."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format=receipt_schema
        )

        # Parse the response
        extracted_json = json.loads(response.choices[0].message.content)
        return extracted_json

    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}")
        raise


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing steps to improve OCR accuracy.

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to handle different lighting conditions
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Noise removal
    gray = cv2.medianBlur(gray, 3)

    # Dilation to make text more prominent
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)

    return gray


def process_image(image_path: str) -> Dict[str, Any]:
    """Process an image file and extract text using OCR.

    Args:
        image_path (str): Path to the image file

    Returns:
        Dict[str, Any]: Dictionary containing OCR results with text
    """
    start_time = time.time()
    logger.info(f"Starting image processing: {image_path}")

    try:
        # Read image using OpenCV
        logger.debug("Reading image with OpenCV")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        logger.debug("Converting BGR to RGB")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing
        logger.debug("Applying image preprocessing")
        processed_image = preprocess_image(image)

        # Perform OCR with improved configuration
        logger.info("Performing OCR on image")
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
        text = pytesseract.image_to_string(processed_image, config=config)

        # Store raw OCR text
        raw_text = text

        # Clean up and format the extracted text
        text_lines = [line.strip()
                      for line in text.split('\n') if line.strip()]
        formatted_text = ' '.join(text_lines)

        # Clean and normalize the text
        cleaned_text = text_processor.clean_text(raw_text)

        # Classify the document
        doc_type, confidence = text_processor.classify_document(cleaned_text)
        logger.info(
            f"Document classified as: {doc_type} with confidence: {confidence:.2f}")

        total_time = time.time() - start_time
        logger.info(f"Image processing completed in {total_time:.2f} seconds")

        # Prepare the response
        response = {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "classification": {
                "type": doc_type,
                "confidence": confidence
            },
            "processing_info": {
                "processing_time": total_time
            }
        }

        # Process with LLM if document is receipt or invoice
        if doc_type.lower() in ['receipt', 'invoice']:
            try:
                structured_data = process_with_llm(cleaned_text)
                response["structured_data"] = structured_data
            except Exception as e:
                logger.error(f"LLM processing failed: {str(e)}")
                response["error"] = f"LLM processing failed: {str(e)}"
        else:
            response["error"] = f"Document type '{doc_type}' is not supported for structured data extraction"

        return response

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise
