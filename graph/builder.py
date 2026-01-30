"""
LangGraph state machine builder for the forecasting agent.
"""

from typing import Any, Sequence

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    research_node,
    agent_node,
    forecast_node,
    tools_node,
    should_continue,
)
from models.base import BaseLangGraphModel


def build_forecast_agent(
    model: BaseLangGraphModel,
    tools: Sequence[Any],
    max_iterations: int = 100,
) -> Any:
    """
    Build the LangGraph state machine for the forecasting agent.

    The agent follows this flow:
    1. research: Set up initial context and question
    2. agent: LLM decides to search or predict
    3. tools: Execute tool calls (if any)
    4. forecast: Extract final prediction

    Graph structure:
        research -> agent -> [tools -> agent]* -> forecast -> END

    Args:
        model: The LangGraph model adapter
        tools: List of tools (e.g., web_search)
        max_iterations: Maximum number of agent iterations

    Returns:
        Compiled LangGraph state machine
    """
    # Create the state graph
    graph = StateGraph(AgentState)

    # Bind tools to the model for the agent node
    model_with_tools = model.bind_tools(tools)

    # Create a model without tools for the forecast node
    # (We want structured output, not tool calls)
    model_for_forecast = model

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("agent", lambda s: agent_node(s, model_with_tools))
    graph.add_node("tools", lambda s: tools_node(s, tools))  # Custom node for search tracking
    graph.add_node("forecast", lambda s: forecast_node(s, model_for_forecast))

    # Set entry point
    graph.set_entry_point("research")

    # Add edges
    graph.add_edge("research", "agent")

    # Conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "forecast": "forecast",
        },
    )

    # Tools always go back to agent
    graph.add_edge("tools", "agent")

    # Forecast ends the graph
    graph.add_edge("forecast", END)

    # Compile and return
    return graph.compile()
