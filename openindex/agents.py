import asyncio

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from pydantic import BaseModel

from openindex.config import TreeConfig
from openindex.models import FlatSectionList, LocateResult, VerifyResult
from openindex.utils.logger import get_logger

# Wiki agents imported lazily to avoid circular imports at module load time

logger = get_logger(__name__)


class AgentPool:
    """Shared pool of agno Agents and a concurrency semaphore.

    All agents use the same underlying OpenAILike model. The semaphore
    limits how many LLM calls run concurrently across all pipeline phases.

    Attributes:
        model_name: model identifier, also used for litellm token counting.
        generator: extracts flat section list from tagged page text.
        verifier: checks if a section title appears on its stated page.
        locator: finds the correct page for a misplaced section.
        summarizer: generates section summaries and doc description.
        wiki_planner: plans which wiki concept pages to create/update/cross-link.
        wiki_concept: generates or rewrites a single wiki concept page.
        sem: asyncio semaphore gating all LLM calls.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        config: TreeConfig,
        extra_body: dict | None = None,
    ):
        self.model_name = model_name
        model = OpenAILike(
            id=model_name, api_key=api_key, base_url=base_url, extra_body=extra_body
        )
        from openindex.wiki.models import ConceptPage, ConceptPlan

        self.generator = Agent(model=model, output_schema=FlatSectionList)
        self.verifier = Agent(model=model, output_schema=VerifyResult)
        self.locator = Agent(model=model, output_schema=LocateResult)
        self.summarizer = Agent(model=model)
        self.wiki_planner = Agent(model=model, output_schema=ConceptPlan)
        self.wiki_concept = Agent(model=model, output_schema=ConceptPage)
        self.sem = asyncio.Semaphore(config.max_parallel_llm_calls)


async def run_structured(
    agent: Agent,
    prompt: str,
    sem: asyncio.Semaphore,
    images: list | None = None,
) -> BaseModel:
    """Run an agent that returns a structured Pydantic output.

    Args:
        agent: agno Agent with output_schema set.
        prompt: user prompt string.
        sem: shared semaphore to limit concurrency.
        images: optional list of image bytes to attach (for vision models).

    Returns:
        Parsed Pydantic model from agent response.
    """
    from agno.media import Image
    async with sem:
        kwargs = {}
        if images:
            kwargs["images"] = [Image(content=b) for b in images]
        result = await agent.arun(prompt, **kwargs)
        return result.content


async def run_text(
    agent: Agent,
    prompt: str,
    sem: asyncio.Semaphore,
) -> str:
    """Run an agent that returns plain text output.

    Args:
        agent: agno Agent without output_schema.
        prompt: user prompt string.
        sem: shared semaphore to limit concurrency.

    Returns:
        Agent response as a string, or empty string on failure.
    """
    async with sem:
        result = await agent.arun(prompt)
        return result.content if result else ""
