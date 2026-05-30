# OpenIndex

## Overview

**OpenIndex** parses PDF documents into a hierarchical section tree and compiles them into a persistent, cross-linked wiki that agents can query.

It combines two projects:

- [PageIndex](https://github.com/VectifyAI/PageIndex) — LLM-based hierarchical section extraction from PDFs
- [OpenKB](https://github.com/VectifyAI/OpenKB) — compiles documents into a queryable wiki with cross-document concept pages

Unlike traditional RAG (which rediscovers knowledge on every query), OpenIndex compiles once: sections are indexed, summaries generated, concept pages created with bidirectional links, and a structured wiki is written to disk. An agent can then search the wiki to answer questions precisely.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Index a PDF](#index-a-pdf)
  - [Query the wiki](#query-the-wiki)
- [License](#license)

## Installation

**From PyPI:**

```bash
pip install openindex
```

**From source:**

```bash
uv pip install git+https://github.com/hienhayho/openindex.git
```

## Usage

Set environment variables (or use a `.env` file):

```
OPENAI_MODEL_NAME=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=
OPENAI_EXTRA_BODY={}
```

> Note: openindex works with any OpenAI-compatible API server (OpenAI, vLLM, Ollama, LM Studio, etc.). Set `OPENAI_BASE_URL` to point to your server.

### Index a PDF

Runs the full pipeline: section extraction → verification → tree building → summaries → wiki generation.

```python
import os
import json
from dotenv import load_dotenv
from openindex import PageIndex, TreeConfig

load_dotenv()

index = PageIndex(
    model_name=os.getenv("OPENAI_MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    extra_body=json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}")),
    config=TreeConfig(max_parallel_llm_calls=8),
)

result = index.build_wiki_sync("paper.pdf", "./wiki")
PageIndex.print_result(result)
```

See [`tools/index.py`](tools/index.py) for a full example.

Output wiki structure:

```
wiki/
├── index.md              # master catalog
├── summaries/<doc>.md    # section tree with page ranges
├── concepts/<slug>.md    # cross-document concept pages
└── sources/<doc>.json    # full per-page text
```

### Query the wiki

The query agent searches the compiled wiki to answer questions, fetching only the relevant pages.

```python
import os
import json
from dotenv import load_dotenv
from openindex import WikiQueryAgent

load_dotenv()

agent = WikiQueryAgent(
    wiki_dir="./wiki",
    model_name=os.getenv("OPENAI_MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    extra_body=json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}")),
)

answer = agent.ask_sync("What is RAG?")
print(answer)
```

See [`tools/query.py`](tools/query.py) for a full example.

## License

MIT
