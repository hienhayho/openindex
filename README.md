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
  - [Return types](#return-types)
  - [Return wiki as dict](#return-wiki-as-dict-not-queryable)
  - [Query the wiki](#query-the-wiki)
  - [Async usage](#async-usage)
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
from openindex import WikiIndex, TreeConfig

load_dotenv()

index = WikiIndex(
    model_name=os.getenv("OPENAI_MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    extra_body=json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}")),
    config=TreeConfig(max_parallel_llm_calls=8),
)

result = index.build_wiki_sync("paper.pdf", "./wiki")  # returns BuildResult
WikiIndex.print_result(result)
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

### Return types

All `build_wiki` / `build_wiki_sync` calls return a `BuildResult` pydantic model:

```python
from openindex import BuildResult, WikiDict
from openindex.models import SectionNode

result: BuildResult = index.build_wiki_sync("paper.pdf", "./wiki")

result.title        # str — document stem, e.g. "paper"
result.doc_name     # str — filename, e.g. "paper.pdf"
result.description  # str — one-paragraph document summary
result.nodes        # list[SectionNode] — nested section tree
result.pages        # dict[int, str] — 1-based page index → page text
result.wiki         # WikiDict — compiled wiki artifacts (always set)
```

`result.wiki` is a `WikiDict`:

```python
result.wiki.doc_name     # str — document stem
result.wiki.description  # str — one-paragraph document summary
result.wiki.summary      # str — full section tree as Markdown
result.wiki.sources      # list[SourcePage] — per-page content
result.wiki.concepts     # dict[str, ConceptEntry] — concept pages keyed by slug
result.wiki.related      # list[str] — related concept slugs (from LLM planner)
result.wiki.index        # str — index.md content for this document
```

`WikiDict` is JSON-serializable via `.model_dump()` or `.model_dump_json()` for database storage.

Sample `BuildResult`:

```python
BuildResult(
    title="paper",
    doc_name="paper.pdf",
    description="This paper introduces a novel attention mechanism for transformer models...",
    nodes=[
        SectionNode(
            title="Introduction",
            start_index=1,
            end_index=3,
            depth=0,
            summary="Introduces the problem of efficient attention in long sequences.",
            children=[
                SectionNode(
                    title="Motivation",
                    start_index=1,
                    end_index=2,
                    depth=1,
                    summary="Existing approaches scale quadratically with sequence length.",
                    children=[],
                ),
            ],
        ),
        SectionNode(title="Method", start_index=4, end_index=8, depth=0, summary="...", children=[]),
    ],
    pages={
        1: "Page 1 text...",
        2: "Page 2 text...",
    },
    wiki=WikiDict(
        doc_name="paper",
        description="This paper introduces a novel attention mechanism...",
        summary="---\ndoc_type: pageindex\nfull_text: sources/paper.json\n---\n\n# Introduction (pages 1–3)\n...",
        sources=[
            SourcePage(page=1, content="Page 1 text...", images=[]),
            SourcePage(page=2, content="Page 2 text...", images=[]),
        ],
        concepts={
            "attention-mechanism": ConceptEntry(
                brief="A mechanism for focusing on relevant parts of the input.",
                content="## Attention Mechanism\n\nAttention allows models to...",
            ),
        },
        related=["transformer"],
        index="# Knowledge Base Index\n\n## Documents\n\n- [[summaries/paper]] (pageindex)...",
    ),
)
```

### Return wiki as dict (Not queryable)

Omit `wiki_dir` to get wiki artifacts in-memory only — nothing written to disk. The result is not queryable by `WikiQueryAgent`. Useful for pipelines that process multiple documents before persisting.

```python
result = index.build_wiki_sync("paper.pdf")  # no wiki_dir
wiki: WikiDict = result.wiki
```

To save to disk later:

```python
from openindex.wiki import save_wiki_dicts_to_dir

wiki1 = index.build_wiki_sync("paper1.pdf").wiki
wiki2 = index.build_wiki_sync("paper2.pdf").wiki

save_wiki_dicts_to_dir([wiki1, wiki2], "./wiki")
```

To build a unified index string across multiple documents:

```python
from openindex.wiki import build_unified_index

wiki1 = index.build_wiki_sync("paper1.pdf").wiki
wiki2 = index.build_wiki_sync("paper2.pdf").wiki

index_md = build_unified_index([wiki1, wiki2])
print(index_md)
```

### Query the wiki

The query agent searches the compiled wiki to answer questions, always fetching actual page content before responding.

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
    cite=True,  # append \source{doc, p.N} after every factual claim
)

answer = agent.ask_sync("What is RAG?")
print(answer)
```

**`cite=True`** (default) — every factual sentence is followed by an inline source marker:

```
RAG combines retrieval with generation to ground LLM outputs. \source{rag-paper, p.3}
Two retrieval strategies are used: sparse (BM25) and dense (DPR). \source{rag-paper, p.4-5}
```

**`cite=False`** — plain answer, no source markers:

```python
agent = WikiQueryAgent(..., cite=False)
answer = agent.ask_sync("What is RAG?")
```

See [`tools/query.py`](tools/query.py) for a full example.

### Async usage

Both `WikiIndex` and `WikiQueryAgent` expose async methods directly. Use these inside an existing event loop (FastAPI, async scripts, etc.) to avoid the overhead of `asyncio.run()`.

**Index:**

```python
import asyncio
import os
import json
from dotenv import load_dotenv
from openindex import WikiIndex, TreeConfig

load_dotenv()

async def main():
    index = WikiIndex(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        extra_body=json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}")),
        config=TreeConfig(max_parallel_llm_calls=8),
    )

    result = await index.build_wiki("paper.pdf", "./wiki")
    WikiIndex.print_result(result)

asyncio.run(main())
```

**Query:**

```python
import asyncio
import os
import json
from dotenv import load_dotenv
from openindex import WikiQueryAgent

load_dotenv()

async def main():
    agent = WikiQueryAgent(
        wiki_dir="./wiki",
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        extra_body=json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}")),
    )

    answer = await agent.ask("What is RAG?")
    print(answer)

asyncio.run(main())
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

This project incorporates code from:
- [PageIndex](https://github.com/VectifyAI/PageIndex) — MIT License
- [OpenKB](https://github.com/VectifyAI/OpenKB) — Apache 2.0 License
