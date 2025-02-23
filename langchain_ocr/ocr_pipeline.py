import argparse
from typing import Dict, Any
from langgraph.graph import Graph, END, START
from ocr_processor import OCRProcessor
from document_classifier import DocumentClassifier
from data_extractor import StructuredDataExtractor
from type import AgentState


class OCRPipeline:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.document_classifier = DocumentClassifier()
        self.data_extractor = StructuredDataExtractor()

    def process_image(self, image_path: str) -> Dict[str, Any]:
        # Initialize state
        state: AgentState = {
            'image_path': image_path,
            'raw_text': '',
            'cleaned_text': '',
            'classification': {},
            'structured_data': {},
            'error': ''
        }

        # Process through pipeline
        state = self.ocr_processor.process_image(state)
        if not state.get('error'):
            state = self.document_classifier.classify_document(state)
            if not state.get('error'):
                state = self.data_extractor.extract_data(state)

        return state


def create_ocr_graph() -> Graph:
    # Initialize processors
    ocr = OCRProcessor()
    classifier = DocumentClassifier()
    extractor = StructuredDataExtractor()

    # Define the workflow
    workflow = Graph()

    # Add nodes
    workflow.add_node("ocr", ocr.process_image)
    workflow.add_node("classify", classifier.classify_document)
    workflow.add_node("extract", extractor.extract_data)

    # Define conditional edges
    def should_extract(state: AgentState) -> str:
        if state.get('error') or state['classification'].get('type') == 'other':
            return END
        return "extract"

    # Define edges with conditions
    workflow.add_edge(START, "ocr")
    workflow.add_edge("ocr", "classify")
    workflow.add_conditional_edges(
        "classify",
        should_extract
    )
    workflow.add_edge("extract", END)

    # Compile the graph
    return workflow.compile()


def process_document(image_path: str) -> Dict[str, Any]:
    # Initialize the graph
    graph = create_ocr_graph()

    # Create initial state
    initial_state: AgentState = {
        "image_path": image_path,
        "raw_text": "",
        "cleaned_text": "",
        "classification": {},
        "structured_data": {},
        "error": ""
    }

    # Execute the graph
    final_state = graph.invoke(initial_state)

    # Return the results
    return {
        "raw_text": final_state["raw_text"],
        "cleaned_text": final_state["cleaned_text"],
        "classification": final_state["classification"],
        "processing_info": {"processing_time": 0},
        "structured_data": final_state.get("structured_data", {}),
        "error": final_state.get("error", "")
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a document image using OCR')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input image file')
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    image_path = args['input_path']
    result = process_document(image_path)
    print(result)
