# DocuSense RAG System

A modern Retrieval-Augmented Generation system built from scratch.

## Features

- Multi-format document support (PDF, DOCX, TXT)
- Vision model integration for image understanding
- Semantic chunking with Markdown awareness
- Free-tier LLMs (Ollama + Gemini)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from docusense import DocumentPipeline

pipeline = DocumentPipeline()
result = pipeline.process_document("document.pdf")
print(f"Created {result.total_chunks} chunks")
```

## License

MIT License
