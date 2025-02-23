from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
from type import AgentState


class StructuredDataExtractor:
    def __init__(self):
        # Define the receipt schema
        self.receipt_schema = {
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

        self.model = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_name="fused_3b",
            model_kwargs={"response_format": self.receipt_schema}
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced receipt processing AI. Extract structured data from receipt text and format it according to the specified JSON schema."),
            ("user", "{text}\n\nDocument type: {doc_type}")
        ])

    def extract_data(self, state: AgentState) -> AgentState:
        try:
            if state.get('error') or state['classification']['type'] == 'other':
                state['error'] = "Document type 'other' is not supported for structured data extraction"
                return state

            response = self.model.invoke(
                self.prompt.format_messages(
                    text=state['cleaned_text'],
                    doc_type=state['classification']['type']
                )
            )

            state['structured_data'] = json.loads(response.content)
            return state
        except Exception as e:
            state['error'] = f"Data extraction failed: {str(e)}"
            return state
