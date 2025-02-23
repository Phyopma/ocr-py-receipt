from setuptools import setup, find_packages

setup(
    name="langchain_ocr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langgraph",
        "langchain-openai",
        "pillow",
        "pytesseract"
    ],
    python_requires=">=3.8",
    description="OCR processing pipeline using LangChain",
    author="Your Name",
    author_email="your.email@example.com"
)