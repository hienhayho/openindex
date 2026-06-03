"""Wiki Q&A agent that answers questions by searching the compiled wiki folder."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from agno.agent import Agent, Message
from agno.models.openai.like import OpenAILike

_QUERY_INSTRUCTIONS = """\
You are a knowledge-base Q&A agent. Answer questions by searching the wiki.

## Wiki layout

  index.md                    — master catalog: every doc and concept with a one-liner
  summaries/{doc}.md          — per-document section tree with page ranges and summaries
  concepts/{slug}.md          — cross-document topic synthesis
  sources/{doc}.json          — full per-page text for pageindex documents (JSON array)

## Tools

### read_file(path)
Read any Markdown file from the wiki. Path is relative to wiki root.

When to use:
- Always start with read_file("index.md") to discover what is available.
- Use read_file("summaries/{doc}.md") to read a document's section-level overview.
  The frontmatter shows doc_type: pageindex and full_text: sources/{doc}.json.
  The body shows headings like "# Section Title (pages 3-7)" — note the page ranges.
- Use read_file("concepts/{slug}.md") to read cross-document topic synthesis.

How to use:
- Pass exact relative path: "index.md", "summaries/paper.md", "concepts/attention.md"
- Do NOT pass absolute paths or paths starting with /

### get_page_content(doc_name, pages)
Fetch specific pages from a pageindex document's full-text JSON source.

When to use:
- ONLY for documents marked doc_type: pageindex in their summary frontmatter.
- When the summary is not enough — you need verbatim text or exact details.
- Target the specific section page range shown in the summary heading.

How to use:
- doc_name: filename stem only, no extension. E.g. for sources/paper.json -> "paper".
- pages: compact range spec. Examples: "3", "3-7", "3-7,12", "1-3,8,10-12".
- Fetch tight ranges (2-5 pages). Never request the entire document.
- If a section spans pages 3-7, call get_page_content("paper", "3-7").

### list_wiki_files(directory)
List all Markdown files in a wiki subdirectory.

When to use:
- When index.md does not clearly show what is available.
- To discover all concept pages: list_wiki_files("concepts").
- To discover all summary pages: list_wiki_files("summaries").

How to use:
- Pass the subdirectory name only: "summaries", "concepts", "explorations".

## Retrieval steps

Follow this order for every question:

1. Catalog — read_file("index.md").
   Identify which documents and concepts are relevant.
   Each document is listed as [[summaries/{doc}]] (pageindex) — brief.

2. Summary — read_file("summaries/{doc}.md") for each relevant document.
   Read section headings and their page ranges. Often enough to answer.
   If the summary links a related concept, read it next.

3. Concepts — read_file("concepts/{slug}.md") for cross-document topics.
   Concept pages synthesize information from multiple documents.

4. Deep dive — get_page_content(doc_name, pages) only when you need details
   beyond the summary. Use the page ranges from the summary headings.
   Prefer the tightest range that covers the relevant section.

5. Discover — if index.md seems incomplete, use list_wiki_files("summaries") or
   list_wiki_files("concepts") to find pages not listed in the index.

6. Answer — synthesize a clear, concise response citing specific documents or sections.
   Reference as: "According to {doc}, section '{title}' (pages N-M), ..."

## Rules

- Answer based ONLY on wiki content. Do not use prior knowledge to fill gaps.
- If information is not in the wiki, say so clearly.
- Before each tool call, output one short sentence explaining what you are looking for.
- Never call get_page_content on a short document (doc_type: short) — use read_file instead.
- Never fetch more pages than needed. Start narrow; expand only if the answer is not there.
"""


def _parse_pages(pages: str) -> list[int]:
    """Parse a page specification string into a sorted list of page numbers.

    Args:
        pages: Page spec like "3-5,7,10-12".

    Returns:
        Sorted list of positive page numbers.
    """
    result: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            segments = part.split("-")
            with contextlib.suppress(ValueError):
                if len(segments) == 2:
                    start, end = int(segments[0]), int(segments[1])
                    result.update(range(start, end + 1))
        else:
            with contextlib.suppress(ValueError):
                result.add(int(part))
    return sorted(n for n in result if n > 0)


class WikiQueryAgent:
    """Q&A agent that searches an OpenKB-compatible wiki to answer questions.

    Uses agno Agent with three tools: read_file, get_page_content, list_wiki_files.
    The wiki must have been produced by WikiIndex.build_wiki() or compile_wiki().

    Attributes:
        wiki_dir: Resolved path to the wiki root directory.

    Example:
        agent = WikiQueryAgent(wiki_dir="./my_wiki", model_name="...",
                               base_url="...", api_key="...")
        answer = agent.ask_sync("What is the main contribution of this paper?")
    """

    def __init__(
        self,
        wiki_dir: str,
        model_name: str,
        base_url: str,
        api_key: str,
        extra_body: dict | None = None,
    ):
        """Initialize the query agent.

        Args:
            wiki_dir: Path to the wiki root directory (must exist).
            model_name: OpenAI-compatible model ID.
            base_url: API base URL.
            api_key: API key.
            extra_body: Optional extra request body fields forwarded to the API.
        """
        self._wiki_dir = Path(wiki_dir).resolve()
        root = self._wiki_dir

        def read_file(path: str) -> str:
            """Read a Markdown file from the wiki.

            Args:
                path: File path relative to wiki root, e.g. 'summaries/paper.md' or 'index.md'.
            """
            full = (root / path).resolve()
            if not full.is_relative_to(root):
                return "Access denied: path escapes wiki root."
            if not full.exists():
                return f"File not found: {path}"
            return full.read_text(encoding="utf-8")

        def get_page_content(doc_name: str, pages: str) -> str:
            """Get text content of specific pages from a pageindex document.

            Only use for documents with doc_type: pageindex. Reads the per-page
            JSON array at sources/{doc_name}.json and returns only the requested pages.

            Args:
                doc_name: Document name without extension, e.g. 'attention-is-all-you-need'.
                pages: Page specification string, e.g. '3-5,7,10-12'.
            """
            target = (root / "sources" / f"{doc_name}.json").resolve()
            if not target.is_relative_to(root):
                return "Access denied: path escapes wiki root."
            if not target.exists():
                return f"File not found: sources/{doc_name}.json"

            data = json.loads(target.read_text(encoding="utf-8"))
            requested = set(_parse_pages(pages))
            matches = [entry for entry in data if entry.get("page") in requested]

            if not matches:
                return f"No content found for pages {pages} in {doc_name}."

            parts: list[str] = []
            for entry in matches:
                page_num = entry["page"]
                content = entry.get("content", "")
                block = f"[Page {page_num}]\n{content}"
                images = entry.get("images")
                if images:
                    paths = ", ".join(img["path"] for img in images if "path" in img)
                    if paths:
                        block += f"\n[Images: {paths}]"
                parts.append(block)

            return "\n\n".join(parts) + "\n\n"

        def list_wiki_files(directory: str) -> str:
            """List all Markdown files in a wiki subdirectory.

            Args:
                directory: Subdirectory relative to wiki root, e.g. 'summaries' or 'concepts'.
            """
            target = (root / directory).resolve()
            if not target.is_relative_to(root):
                return "Access denied: path escapes wiki root."
            if not target.exists() or not target.is_dir():
                return "No files found."
            md_files = sorted(p.name for p in target.iterdir() if p.suffix == ".md")
            return "\n".join(md_files) if md_files else "No files found."

        self.read_file = read_file

        model = OpenAILike(
            id=model_name,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
        )
        self._agent = Agent(
            model=model,
            tools=[read_file, get_page_content, list_wiki_files],
            instructions=_QUERY_INSTRUCTIONS,
            debug_mode=True,
        )

    async def ask(self, question: str) -> str:
        """Answer a question by searching the wiki.

        Args:
            question: Natural language question about the indexed documents.

        Returns:
            Answer string grounded in wiki content.
        """
        messages = [
            Message(role="user", content=question),
            Message(
                role="tool",
                content=self.read_file("index.md"),
                tool_args={"path": "index.md"},
                tool_name="read_file",
            ),
        ]
        result = await self._agent.arun(input=messages)
        return result.content or ""

    def ask_sync(self, question: str) -> str:
        """Synchronous wrapper around ask().

        Args:
            question: Natural language question about the indexed documents.

        Returns:
            Answer string grounded in wiki content.
        """
        return asyncio.run(self.ask(question))
