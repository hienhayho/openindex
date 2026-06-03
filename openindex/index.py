import asyncio
from pathlib import Path

from openindex.agents import AgentPool
from openindex.wiki.models import BuildResult
from openindex.config import TreeConfig
from openindex.parser import load_pages, page_to_text
from openindex.pipeline.expander import expand_large_nodes
from openindex.pipeline.generator import generate_flat_sections
from openindex.pipeline.summarizer import add_summaries, generate_doc_description
from openindex.pipeline.tree_builder import add_preface_if_needed, flat_to_tree
from openindex.pipeline.verifier import verify_and_fix
from openindex.utils.logger import get_logger

logger = get_logger(__name__)


class WikiIndex:
    """Entry point for building a hierarchical section index from a PDF.

    Runs a multi-phase pipeline:
      1. Generate flat section list with page assignments
      2. Verify and fix incorrect page assignments
      3. Convert flat list to nested tree
      4. Expand large leaf nodes recursively
      5. Generate per-node summaries and accumulated doc description

    Usage:
        index = WikiIndex(model_name, base_url, api_key)
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

    async def build(
        self,
        pdf_path: str | None = None,
        *,
        texts: list[str] | None = None,
        doc_name: str | None = None,
    ) -> BuildResult:
        """Run the full pipeline and return the structured index.

        Provide either pdf_path (loads and parses the PDF) or texts (pre-extracted
        page strings, skipping PDF loading). When using texts, doc_name is used for
        the result title/doc_name fields; defaults to "document" if omitted.

        Args:
            pdf_path: path to the PDF file. Mutually exclusive with texts.
            texts: pre-extracted page text strings, one entry per page.
            doc_name: document name used in result when texts is provided.

        Returns:
            Dict with keys:
              - title: filename without extension
              - doc_name: filename with extension
              - description: accumulated document description (up to 5 sentences)
              - nodes: list of nested section dicts (see SectionNode.to_dict)
              - pages: dict mapping 1-based page index to page text
        """
        if texts is not None:
            logger.info("using_provided_texts", count=len(texts))
            page_texts = texts
            name = doc_name or "document"
            p = Path(name)
        elif pdf_path is not None:
            logger.info("loading_pdf", path=pdf_path)
            pages = load_pages(pdf_path)
            page_texts = [page_to_text(pg) for pg in pages]
            p = Path(pdf_path)
        else:
            raise ValueError("Provide either pdf_path or texts.")

        total_pages = len(page_texts)
        start_index = 1
        logger.info("pages_ready", pages=total_pages)

        sections = await generate_flat_sections(
            page_texts, self._pool, self._config, start_index
        )
        logger.info("sections_generated", count=len(sections))

        if not sections:
            from openindex.models import FlatSection

            logger.warning("no_sections_found_using_fallback", doc=p.stem)
            sections = [
                FlatSection(structure="1", title=p.stem, physical_index=start_index)
            ]

        sections = add_preface_if_needed(sections, start_index)

        sections = await verify_and_fix(
            sections, page_texts, self._pool, self._config, start_index
        )
        logger.info("sections_verified", count=len(sections))

        nodes = flat_to_tree(sections, total_pages, start_index)
        logger.info("tree_built", roots=len(nodes))

        nodes = await expand_large_nodes(
            nodes, page_texts, self._pool, self._config, start_index
        )
        logger.info("nodes_expanded")

        nodes = await add_summaries(
            nodes, page_texts, self._pool, self._config, start_index
        )
        description = await generate_doc_description(nodes, self._pool)
        logger.info("summaries_done")

        return BuildResult(
            title=p.stem,
            doc_name=p.name,
            description=description,
            nodes=nodes,
            pages={start_index + i: text for i, text in enumerate(page_texts)},
        )

    def build_sync(
        self,
        pdf_path: str | None = None,
        *,
        texts: list[str] | None = None,
        doc_name: str | None = None,
    ) -> BuildResult:
        """Synchronous wrapper around build().

        Args:
            pdf_path: path to the PDF file.
            texts: pre-extracted page text strings.
            doc_name: document name used in result when texts is provided.

        Returns:
            Same dict as build().
        """
        return asyncio.run(self.build(pdf_path, texts=texts, doc_name=doc_name))

    async def build_wiki(
        self,
        pdf_path: str | None = None,
        wiki_dir: str | None = None,
        *,
        texts: list[str] | None = None,
        doc_name: str | None = None,
    ) -> BuildResult:
        """Run the full pipeline and compile wiki artifacts.

        Calls build() then compiles wiki artifacts: sources JSON, summary
        Markdown, concept pages, backlinks, and index.md.

        Always sets result["wiki"] as a WikiDict. When wiki_dir is provided,
        artifacts are also written to disk (with LLM-based concept merging
        against existing concepts). When wiki_dir is None, artifacts are
        in-memory only.

        Args:
            pdf_path: path to the PDF file. Mutually exclusive with texts.
            wiki_dir: path to the wiki root directory (created if missing).
                      Pass None to return wiki artifacts in the result dict.
            texts: pre-extracted page text strings, skips PDF loading.
            doc_name: document name used in result when texts is provided.

        Returns:
            build() dict, with "wiki" key added when wiki_dir is None.
        """
        from openindex.wiki.compiler import compile_wiki

        result = await self.build(pdf_path, texts=texts, doc_name=doc_name)
        result.wiki = await compile_wiki(
            result.model_dump(),
            self._pool,
            wiki_dir=Path(wiki_dir) if wiki_dir else None,
        )
        return result

    def build_wiki_sync(
        self,
        pdf_path: str | None = None,
        wiki_dir: str | None = None,
        *,
        texts: list[str] | None = None,
        doc_name: str | None = None,
    ) -> BuildResult:
        """Synchronous wrapper around build_wiki().

        Args:
            pdf_path: path to the PDF file.
            wiki_dir: path to the wiki root directory. None returns wiki in result dict.
            texts: pre-extracted page text strings.
            doc_name: document name used in result when texts is provided.

        Returns:
            Same dict as build_wiki().
        """
        return asyncio.run(
            self.build_wiki(pdf_path, wiki_dir, texts=texts, doc_name=doc_name)
        )

    @staticmethod
    def print_result(result: BuildResult) -> None:
        """Print a BuildResult in a human-readable tree format.

        Prints title, page count, description, and the full section tree
        with indented headings and page ranges.

        Args:
            result: BuildResult returned by build() or build_wiki().
        """
        title = result.title
        description = result.description
        pages = result.pages
        nodes_data = result.nodes

        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"Pages : {len(pages)}")
        if description:
            print(f"\nDescription:\n  {description}\n")
        print("Structure:")

        def _print_node(node: dict, indent: int = 0) -> None:
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
