"""Wiki Q&A agent that answers questions by searching the compiled wiki folder."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path

from agno.agent import Agent, Message
from agno.models.openai.like import OpenAILike

_WIKI_LAYOUT = """\
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
- Always fetch actual page content before answering — summaries alone are not enough.
- Target the specific section page range shown in the summary heading.

How to use:
- doc_name: the stem from the wikilink in index.md.
  index.md lists documents as: [[summaries/{doc_name}]] (pageindex) — brief
  summaries/{doc_name}.md also shows: full_text: sources/{doc_name}.json in frontmatter.
  Use that {doc_name} directly — no extension, no path prefix.
  Example: [[summaries/paper]] → doc_name = "paper"
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

1. Catalog — read_file("index.md"). Identify which documents and concepts are relevant.
2. Summary — read_file("summaries/{doc}.md") for each relevant document.
   Read section headings and their page ranges to find the relevant sections.
3. Concepts — read_file("concepts/{slug}.md") for cross-document topics if relevant.
4. Page content — ALWAYS call get_page_content(doc_name, pages) for the relevant
   sections identified in the summary. Use the exact page ranges from the summary headings.
   Do not answer from summaries alone — always fetch the actual page text.
   Prefer the tightest range that covers the relevant section; expand if needed.
5. Discover — if index.md seems incomplete, use list_wiki_files("summaries") or
   list_wiki_files("concepts") to find pages not listed in the index.
"""

_ANSWER_NO_CITE = """\
6. Answer — output ONLY the final answer. Do not include any preamble, narration,
   reasoning steps, or explanation of what you searched. Start directly with the answer content.
   Do NOT include any source citations or references in the answer.

## Rules

- Answer based ONLY on wiki content. Do not use prior knowledge to fill gaps.
- If information is not in the wiki, say so clearly.
- ALWAYS fetch page content with get_page_content before answering.
- Do NOT add any citations, source markers, or page references in the answer.
- Do NOT narrate your retrieval process ("I will now check...", "Based on the summary...").
- Output ONLY the final answer text. Nothing before it.
- Never call get_page_content on a short document (doc_type: short) — use read_file instead.
- Never fetch more pages than needed. Start narrow; expand only if the answer is not there.
"""

_ANSWER_WITH_CITE = """\
6. Answer — output ONLY the final answer. Do not include any preamble, narration,
   reasoning steps, or explanation of what you searched. Start directly with the answer content.
   After EVERY sentence or clause that states a fact, append a source marker:
     \\source{doc_name, p.N}       — single page
     \\source{doc_name, p.N-M}     — page range
     \\source{doc_name, p.N, p.M}  — non-consecutive pages
   Example:
     "The attention mechanism scales quadratically. \\source{paper, p.3}
      Two variants exist: additive and dot-product. \\source{paper, p.4-5}"
   If a sentence draws from multiple documents:
     "This appears in both datasets. \\source{paper1, p.2} \\source{paper2, p.7}"
   Do NOT group all sources at the end — every claim must have its marker inline.

## Rules

- Answer based ONLY on wiki content. Do not use prior knowledge to fill gaps.
- If information is not in the wiki, say so clearly.
- ALWAYS fetch page content with get_page_content before answering.
- Every factual sentence MUST be followed by a \\source{} marker. No exceptions.
- You MAY ONLY cite a page you have actually fetched via get_page_content in this session.
  NEVER invent or guess page numbers. If you have not fetched a page, you cannot cite it.
  If you cannot cite a claim, do not make it.
- Do NOT narrate your retrieval process ("I will now check...", "Based on the summary...").
- Output ONLY the final answer text. Nothing before it.
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
        cite: bool = True,
    ):
        """Initialize the query agent.

        Args:
            wiki_dir: Path to the wiki root directory (must exist).
            model_name: OpenAI-compatible model ID.
            base_url: API base URL.
            api_key: API key.
            extra_body: Optional extra request body fields forwarded to the API.
            cite: If True, agent appends \\source{doc, p.N} markers after every
                  factual claim. If False, answer has no citations. Default True.
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

        self._read_file = read_file
        self._cite = cite

        model = OpenAILike(
            id=model_name,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
        )
        instructions = _WIKI_LAYOUT + (_ANSWER_WITH_CITE if cite else _ANSWER_NO_CITE)
        self._agent = Agent(
            model=model,
            tools=[read_file, get_page_content, list_wiki_files],
            instructions=instructions,
            debug_mode=True,
        )

    @staticmethod
    def _fetched_pages(tools) -> dict[str, set[int]]:
        """Build map of doc_name → set of page numbers actually fetched via get_page_content."""
        fetched: dict[str, set[int]] = {}
        for t in (tools or []):
            if t.tool_name != "get_page_content":
                continue
            args = t.tool_args or {}
            doc = args.get("doc_name", "")
            pages_str = args.get("pages", "")
            if doc and pages_str:
                fetched.setdefault(doc, set()).update(_parse_pages(pages_str))
        return fetched

    @staticmethod
    def _cited_pages(answer: str) -> list[tuple[str, list[int]]]:
        """Extract (doc_name, pages) pairs from \\source{} markers in answer."""
        cited: list[tuple[str, list[int]]] = []
        for m in re.finditer(r"\\source\{([^}]+)\}", answer):
            parts = [p.strip() for p in m.group(1).split(",")]
            if not parts:
                continue
            doc = parts[0]
            pages: list[int] = []
            for part in parts[1:]:
                part = part.strip()
                if part.startswith("p."):
                    part = part[2:]
                pages.extend(_parse_pages(part))
            if doc and pages:
                cited.append((doc, pages))
        return cited

    @staticmethod
    def _unfetched_citations(
        cited: list[tuple[str, list[int]]],
        fetched: dict[str, set[int]],
    ) -> list[str]:
        """Return list of citation strings that reference pages not actually fetched."""
        bad: list[str] = []
        for doc, pages in cited:
            fetched_for_doc = fetched.get(doc, set())
            missing = [p for p in pages if p not in fetched_for_doc]
            if missing:
                pages_str = ",".join(str(p) for p in missing)
                bad.append(f"{doc} p.{pages_str}")
        return bad

    async def ask(self, question: str) -> str:
        """Answer a question by searching the wiki.

        When cite=True, verifies that every \\source{} marker in the answer
        corresponds to a page actually fetched via get_page_content. If hallucinated
        citations are found, prompts the agent to re-read those pages and revise.

        Args:
            question: Natural language question about the indexed documents.

        Returns:
            Answer string grounded in wiki content.
        """
        messages = [
            Message(role="user", content=question),
            Message(
                role="tool",
                content=self._read_file("index.md"),
                tool_args={"path": "index.md"},
                tool_name="read_file",
            ),
        ]
        result = await self._agent.arun(input=messages)
        answer = result.content or ""

        if not self._cite:
            return answer

        fetched = self._fetched_pages(result.tools)
        cited = self._cited_pages(answer)
        bad = self._unfetched_citations(cited, fetched)

        if bad:
            bad_list = ", ".join(bad)
            retry_prompt = (
                f"Your previous answer cited pages you never fetched: {bad_list}.\n"
                f"Steps:\n"
                f"1. Call get_page_content for each of those pages now.\n"
                f"2. Answer the original question using ONLY pages you have actually fetched.\n"
                f"3. Output ONLY the final answer — no preamble, no narration.\n\n"
                f"Original question: {question}"
            )
            retry_result = await self._agent.arun(retry_prompt)
            answer = retry_result.content or answer

        return answer

    def ask_sync(self, question: str) -> str:
        """Synchronous wrapper around ask().

        Args:
            question: Natural language question about the indexed documents.

        Returns:
            Answer string grounded in wiki content.
        """
        return asyncio.run(self.ask(question))
