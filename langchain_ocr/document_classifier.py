from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
from type import AgentState


class DocumentClassifier:
    def __init__(self):
        # Define the classification schema
        self.classification_schema = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["receipt", "invoice", "other"],
                    "description": "The type of document"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score between 0 and 1"
                }
            },
            "required": ["type", "confidence"]
        }

        self.model = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_name="lmstudio-community/llama-3.2-1b-instruct",
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification",
                        "schema": self.classification_schema
                    }
                }
            }
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a document classification expert. Analyze the given text and classify it into one of these categories: 'receipt', 'invoice', 'other'. Return a JSON object with the document type and your confidence level in the classification."),
            ("user", "{text}")
        ])

    def classify_document(self, state: AgentState) -> AgentState:
        try:
            if state.get('error'):
                return state

            response = self.model.invoke(
                self.prompt.format_messages(
                    text=state['cleaned_text'],
                    schema=json.dumps(self.classification_schema, indent=2)
                )
            )

            # Parse JSON response directly without using eval
            state['classification'] = json.loads(response.content)
            return state
        except Exception as e:
            state['error'] = f"Classification failed: {str(e)}"
            return state
