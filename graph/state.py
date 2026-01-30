"""
Agent state definition for the forecasting agent.
"""

from typing import Annotated, TypedDict, Optional
from operator import add

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State for the forecasting agent.

    Attributes:
        messages: List of conversation messages (accumulates with add operator)
        question: The prediction market question to answer
        context: Additional context about the market
        yes_means: What a "yes" prediction means
        no_means: What a "no" prediction means
        sample_date: The date of the sample (for context)
        market_price: Current market implied probability
        tools_enabled: Whether web search tools are available
        iterations: Number of agent iterations so far
        max_iterations: Maximum allowed iterations
        prediction: Final prediction ("yes" or "no")
        confidence: Confidence in the prediction (0-100)
        reasoning: Reasoning for the prediction
        search_queries: List of search queries made
        search_results: List of search results obtained
    """

    # Message history (accumulates)
    messages: Annotated[list[BaseMessage], add]

    # Market information (static)
    question: str
    context: str
    yes_means: str
    no_means: str
    sample_date: str
    market_price: Optional[float]

    # Configuration
    tools_enabled: bool

    # Iteration tracking
    iterations: int
    max_iterations: int

    # Output fields
    prediction: Optional[str]
    confidence: Optional[float]
    reasoning: Optional[str]

    # Research tracking
    search_queries: Annotated[list[str], add]
    search_results: Annotated[list[str], add]


def create_initial_state(
    question: str,
    context: str,
    yes_means: str,
    no_means: str,
    sample_date: str,
    market_price: Optional[float] = None,
    max_iterations: int = 100,
    tools_enabled: bool = True,
) -> AgentState:
    """
    Create the initial state for the forecasting agent.

    Args:
        question: The prediction market question
        context: Additional context about the market
        yes_means: What a "yes" prediction means
        no_means: What a "no" prediction means
        sample_date: The date of the sample (YYYY-MM-DD format)
        market_price: Current market implied probability
        max_iterations: Maximum number of agent iterations
        tools_enabled: Whether web search tools are available (default True)

    Returns:
        Initial AgentState
    """
    return AgentState(
        messages=[],
        question=question,
        context=context,
        yes_means=yes_means,
        no_means=no_means,
        sample_date=sample_date,
        market_price=market_price,
        tools_enabled=tools_enabled,
        iterations=0,
        max_iterations=max_iterations,
        prediction=None,
        confidence=None,
        reasoning=None,
        search_queries=[],
        search_results=[],
    )
