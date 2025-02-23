import argparse
import os
import json
import logging
from process_document import process_document, process_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='OCR Processing Tool for Documents (PDFs and Images)'
    )
    
    # Add arguments
    parser.add_argument(
        'input_path',
        help='Path to input file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory path (default: output)'
    )
    parser.add_argument(
        '-d', '--dpi',
        type=int,
        default=300,
        help='DPI for PDF conversion (default: 300)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"Starting OCR processing with arguments: {vars(args)}")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output, exist_ok=True)
        logger.debug(f"Created/verified output directory: {args.output}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        return
    
    # Process input path
    if os.path.isfile(args.input_path):
        # Process single file
        try:
            logger.info(f"Processing single file: {args.input_path}")
            result = process_document(args.input_path, args.dpi)
            output_file = os.path.join(
                args.output,
                f"{os.path.splitext(os.path.basename(args.input_path))[0]}.json"
            )
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Successfully processed: {args.input_path}")
            logger.info(f"Output saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error processing {args.input_path}: {str(e)}")
    
    elif os.path.isdir(args.input_path):
        # Process directory
        try:
            logger.info(f"Processing directory: {args.input_path}")
            process_folder(args.input_path, args.output, args.dpi)
            logger.info(f"Successfully processed all files in: {args.input_path}")
            logger.info(f"Results saved in: {args.output}")
        except Exception as e:
            logger.error(f"Error processing directory {args.input_path}: {str(e)}")
    
    else:
        logger.error(f"Error: {args.input_path} is not a valid file or directory")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        raise