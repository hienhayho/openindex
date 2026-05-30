import asyncio
from pathlib import Path

from openindex.agents import AgentPool
from openindex.config import TreeConfig
from openindex.parser import load_pages, page_to_text
from openindex.pipeline.expander import expand_large_nodes
from openindex.pipeline.generator import generate_flat_sections
from openindex.pipeline.summarizer import add_summaries, generate_doc_description
from openindex.pipeline.tree_builder import add_preface_if_needed, flat_to_tree
from openindex.pipeline.verifier import verify_and_fix
from openindex.utils.logger import get_logger

logger = get_logger(__name__)


class PageIndex:
    """Entry point for building a hierarchical section index from a PDF.

    Runs a multi-phase pipeline:
      1. Generate flat section list with page assignments
      2. Verify and fix incorrect page assignments
      3. Convert flat list to nested tree
      4. Expand large leaf nodes recursively
      5. Generate per-node summaries and accumulated doc description

    Usage:
        index = PageIndex(model_name, base_url, api_key)
        result = index.build_sync("paper.pdf")
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        config: TreeConfig | None = None,
        extra_body: dict | None = None,
    ):
        """Initialize the pipeline with model credentials and config.

        Args:
            model_name: OpenAI-compatible model ID.
            base_url: API base URL.
            api_key: API key.
            config: optional TreeConfig; defaults to TreeConfig() if not provided.
            extra_body: optional extra request body fields forwarded to the API.
        """
        self._config = config or TreeConfig()
        self._pool = AgentPool(model_name, base_url, api_key, self._config, extra_body)

    async def build(self, pdf_path: str) -> dict:
        """Run the full pipeline and return the structured index.

        Args:
            pdf_path: path to the PDF file.

        Returns:
            Dict with keys:
              - title: filename without extension
              - doc_name: filename with extension
              - description: accumulated document description (up to 5 sentences)
              - nodes: list of nested section dicts (see SectionNode.to_dict)
              - pages: dict mapping 1-based page index to page text
        """
        logger.info("loading_pdf", path=pdf_path)
        pages = load_pages(pdf_path)
        texts = [page_to_text(p) for p in pages]
        total_pages = len(texts)
        start_index = 1
        logger.info("pdf_loaded", pages=total_pages)

        sections = await generate_flat_sections(texts, self._pool, self._config, start_index)
        logger.info("sections_generated", count=len(sections))

        sections = add_preface_if_needed(sections, start_index)

        sections = await verify_and_fix(sections, texts, self._pool, self._config, start_index)
        logger.info("sections_verified", count=len(sections))

        nodes = flat_to_tree(sections, total_pages, start_index)
        logger.info("tree_built", roots=len(nodes))

        nodes = await expand_large_nodes(nodes, texts, self._pool, self._config, start_index)
        logger.info("nodes_expanded")

        nodes = await add_summaries(nodes, texts, self._pool, self._config, start_index)
        description = await generate_doc_description(nodes, self._pool)
        logger.info("summaries_done")

        p = Path(pdf_path)
        return {
            "title": p.stem,
            "doc_name": p.name,
            "description": description,
            "nodes": [n.to_dict() for n in nodes],
            "pages": {start_index + i: text for i, text in enumerate(texts)},
        }

    def build_sync(self, pdf_path: str) -> dict:
        """Synchronous wrapper around build().

        Args:
            pdf_path: path to the PDF file.

        Returns:
            Same dict as build().
        """
        return asyncio.run(self.build(pdf_path))

    async def build_wiki(self, pdf_path: str, wiki_dir: str) -> dict:
        """Run the full pipeline and write an OpenKB-compatible wiki folder.

        Calls build() then compiles wiki artifacts: sources JSON, summary
        Markdown, concept pages, backlinks, and index.md.

        Args:
            pdf_path: path to the PDF file.
            wiki_dir: path to the wiki root directory (created if missing).

        Returns:
            Same dict as build().
        """
        from openindex.wiki.compiler import compile_wiki

        result = await self.build(pdf_path)
        await compile_wiki(result, Path(wiki_dir), self._pool)
        return result

    def build_wiki_sync(self, pdf_path: str, wiki_dir: str) -> dict:
        """Synchronous wrapper around build_wiki().

        Args:
            pdf_path: path to the PDF file.
            wiki_dir: path to the wiki root directory.

        Returns:
            Same dict as build().
        """
        return asyncio.run(self.build_wiki(pdf_path, wiki_dir))

    @staticmethod
    def print_result(result: dict) -> None:
        """Print a build() result in a human-readable tree format.

        Prints title, page count, description, and the full section tree
        with indented headings and page ranges.

        Args:
            result: dict returned by build() or build_wiki().
        """
        title = result.get("title", "")
        description = result.get("description", "")
        pages = result.get("pages", {})
        nodes_data = result.get("nodes", [])

        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"Pages : {len(pages)}")
        if description:
            print(f"\nDescription:\n  {description}\n")
        print("Structure:")

        def _print_node(node: dict, indent: int = 0) -> None:
            prefix = "  " * indent
            marker = "│  " * max(indent - 1, 0) + ("├─ " if indent > 0 else "")
            depth = node.get("depth", 0)
            hashes = "#" * (depth + 1)
            title_str = node.get("title", "")
            start = node.get("start_index", "")
            end = node.get("end_index", "")
            summary = node.get("summary", "")
            print(f"  {marker}{hashes} {title_str}  (p.{start}–{end})")
            if summary:
                summary_line = summary[:120] + ("…" if len(summary) > 120 else "")
                print(f"  {'  ' * indent}   {summary_line}")
            for child in node.get("children", []):
                _print_node(child, indent + 1)

        for node in nodes_data:
            _print_node(node, indent=0)
        print(f"{'=' * 60}")
