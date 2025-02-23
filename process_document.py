import os
import json
from typing import Dict, List, Any
from image_processor import process_image
from pdf_processor import process_pdf

def process_document(file_path: str, dpi: int = 300) -> Dict[str, Any]:
    """Process a single document (PDF or image) and return the OCR results.

    Args:
        file_path (str): Path to the input file
        dpi (int, optional): DPI for PDF conversion. Defaults to 300.

    Returns:
        Dict[str, Any]: OCR results containing text and bounding boxes
    """
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Process based on file type
    if ext == '.pdf':
        return process_pdf(file_path, dpi)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return process_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_folder(input_dir: str, output_dir: str, dpi: int = 300) -> None:
    """Process all supported documents in a folder.

    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        dpi (int, optional): DPI for PDF conversion. Defaults to 300.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            try:
                # Process the document
                result = process_document(file_path, dpi)
                
                # Save results to JSON file
                output_file = os.path.join(
                    output_dir,
                    f"{os.path.splitext(filename)[0]}.json"
                )
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=4)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")