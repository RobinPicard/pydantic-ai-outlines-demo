# Pydantic AI Outlines Demo

This repository demonstrates how to use Outlines from Dottxt in Pydantic AI `pydantic-ai` with a multimodal model (Qwen2-VL) through Transformers to extract structured data from receipt images.

## Prerequisites

- Python 3.10 – 3.12
- A GPU with sufficient memory for `Qwen/Qwen2-VL-7B-Instruct`

## Installation

To run the example, create a virtual environment and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The editable install pulls the required packages listed in `pyproject.toml`, including `pydantic-ai-slim[outlines-transformers]` and `torchvision`.

## Running the Demo

```bash
python main.py
```

The script will:

- Download the Qwen2-VL model and processor if they are not already cached
- Instantiate a `pydantic_ai.Agent` with the Pydantic AI Outlines model
- Send a prompt and a sample Trader Joe's receipt image to the model
- Parse the structured output into a `ReceiptSummary` object and print the extracted fields

If you want to test a different receipt image or prompt, update the `IMAGE_PATH` or `PROMPT` constants near the top of `main.py`.
