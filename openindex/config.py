from pydantic import BaseModel, Field


class TreeConfig(BaseModel):
    """Configuration for the PageIndex pipeline.

    Controls token budgets, concurrency, and expansion thresholds.
    All values can be overridden at construction time.

    Attributes:
        max_tokens_per_group: Maximum tokens per page group sent to the generator LLM in one call.
        page_overlap: Number of pages repeated at the start of each group for cross-boundary context.
        max_parallel_llm_calls: Semaphore limit — maximum concurrent LLM calls across all pipeline phases.
        max_pages_per_node: Leaf nodes spanning more pages than this are candidates for recursive expansion.
        max_tokens_per_node: Leaf nodes must also exceed this token count to trigger expansion (both thresholds must be met).
    """

    max_tokens_per_group: int = Field(default=20000)
    page_overlap: int = Field(default=1)
    max_parallel_llm_calls: int = Field(default=5)
    max_pages_per_node: int = Field(default=10)
    max_tokens_per_node: int = Field(default=50000)
