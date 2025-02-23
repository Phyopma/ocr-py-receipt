from typing import Dict, Any, TypedDict

class AgentState(TypedDict):
    image_path: str
    raw_text: str
    cleaned_text: str
    classification: Dict[str, Any]
    structured_data: Dict[str, Any]
    error: str