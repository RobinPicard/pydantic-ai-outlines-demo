# Pydantic AI Outlines Demo

This repository demonstrates how to use [Outlines](https://github.com/dottxt-ai/outlines) from [Dottxt](https://dottxt.ai/) in [Pydantic AI](https://ai.pydantic.dev/) with a multimodal model (Qwen2-VL) through Transformers to extract structured data from receipt images.

## Prerequisites

- Python 3.10 – 3.12
- A GPU with sufficient memory for `Qwen/Qwen2-VL-7B-Instruct`
- `uv` installed

## Installation

To run the example, start by setting up the environment with `uv`

```bash
uv venv
uv sync
```

These commands pull the required packages listed in `pyproject.toml` and defined in `uv.lock`, including `pydantic-ai-slim[outlines-transformers]` and `torchvision`.

## Running the Demo

```bash
uv run main.py
```

The script will:

- Download the Qwen2-VL model and processor if they are not already cached
- Instantiate a `pydantic_ai.Agent` with the Pydantic AI Outlines model
- Send a prompt and a sample Trader Joe's receipt image to the model
- Parse the structured output into a `ReceiptSummary` object and print the extracted fields

If you want to test a different receipt image or prompt, update the `IMAGE_PATH` or `PROMPT` constants near the top of `main.py`.
