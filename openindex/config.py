from pydantic import BaseModel, Field


class TreeConfig(BaseModel):
    """Configuration for the WikiIndex pipeline.

    Controls token budgets, concurrency, and expansion thresholds.
    All values can be overridden at construction time.

    Attributes:
        max_tokens_per_group: Maximum tokens per page group sent to the generator LLM in one call.
        page_overlap: Number of pages repeated at the start of each group for cross-boundary context.
        max_parallel_llm_calls: Semaphore limit — maximum concurrent LLM calls across all pipeline phases.
        max_pages_per_node: Leaf nodes spanning more pages than this are candidates for recursive expansion.
        max_tokens_per_node: Leaf nodes must also exceed this token count to trigger expansion (both thresholds must be met).
        max_fix_attempts: Maximum verify-fix loop iterations during the verification phase.
        fix_search_radius: Pages to search on each side of the stated page when locating a misplaced section.
        max_expansion_depth: Hard recursion cap for expand_large_nodes to prevent runaway expansion.
        max_sections: Maximum total sections to extract across the entire document. 0 means unlimited.
            When set, each page group receives a proportional budget (max_sections // num_groups).
    """

    max_tokens_per_group: int = Field(default=20000)
    page_overlap: int = Field(default=1)
    max_parallel_llm_calls: int = Field(default=5)
    max_pages_per_node: int = Field(default=10)
    max_tokens_per_node: int = Field(default=50000)
    max_fix_attempts: int = Field(default=3)
    fix_search_radius: int = Field(default=5)
    max_expansion_depth: int = Field(default=5)
    max_sections: int = Field(default=0)
