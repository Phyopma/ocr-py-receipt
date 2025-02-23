from typing import Dict, List, Any
from pdf2image import convert_from_path
from image_processor import process_image
import tempfile
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pdf(pdf_path: str, dpi: int = 300) -> Dict[str, Any]:
    """Process a PDF file by converting it to images and performing OCR.

    Args:
        pdf_path (str): Path to the PDF file
        dpi (int, optional): DPI for PDF conversion. Defaults to 300.

    Returns:
        Dict[str, Any]: Combined OCR results from all pages
    """
    start_time = time.time()
    logger.info(f"Starting PDF processing: {pdf_path} with DPI={dpi}")
    
    try:
        # Convert PDF to images
        logger.info("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Successfully converted PDF to {len(images)} images")
        
        # Create temporary directory for saving images
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.debug(f"Created temporary directory: {temp_dir}")
            all_texts = []
            all_boxes = []
            
            # Process each page
            for i, image in enumerate(images):
                page_start_time = time.time()
                logger.info(f"Processing page {i+1}/{len(images)}")
                
                # Save image temporarily
                temp_image_path = os.path.join(temp_dir, f'page_{i+1}.png')
                image.save(temp_image_path, 'PNG')
                logger.debug(f"Saved temporary image: {temp_image_path}")
                
                try:
                    # Process the image
                    result = process_image(temp_image_path)
                    
                    # Append results
                    if result['text']:
                        all_texts.append(f"Page {i+1}:\n{result['text']}")
                        logger.debug(f"Extracted {len(result['text'])} characters from page {i+1}")
                    else:
                        logger.warning(f"No text extracted from page {i+1}")
                    
                    # Add page number to boxes
                    page_boxes = result['boxes']
                    for box in page_boxes:
                        box['page'] = i + 1
                    all_boxes.extend(page_boxes)
                    logger.debug(f"Found {len(page_boxes)} text boxes in page {i+1}")
                    
                    page_time = time.time() - page_start_time
                    logger.info(f"Completed page {i+1} processing in {page_time:.2f} seconds")
                    
                except Exception as e:
                    logger.error(f"Error processing page {i+1}: {str(e)}")
                    raise
        
        total_time = time.time() - start_time
        logger.info(f"PDF processing completed in {total_time:.2f} seconds")
        
        return {
            "text": "\n\n".join(all_texts),
            "boxes": all_boxes
        }
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise