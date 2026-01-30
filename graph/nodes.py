"""
Node functions for the forecasting agent graph.
"""

import re
import logging
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import AgentState
from prompts.templates import AgenticPromptTemplates

logger = logging.getLogger(__name__)


def _get_text_content(content: str | list | None) -> str:
    """
    Extract text from content which may be a string or list of content blocks.

    Anthropic models can return content as a list of content blocks (e.g., when
    there are tool calls or multi-part responses). This helper safely extracts
    the text content as a string.

    Args:
        content: The content field from an AIMessage (str, list, or None)

    Returns:
        The text content as a string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)
    return str(content)


def research_node(state: AgentState) -> dict[str, Any]:
    """
    Initial research node that sets up the agent with the forecasting task.

    This node creates the initial system message and user message with
    the prediction market question and context.

    Args:
        state: Current agent state

    Returns:
        State update with initial messages
    """
    # Check if tools are enabled (default to True for backwards compatibility)
    tools_enabled = state.get("tools_enabled", True)

    # Create system message with forecasting instructions (conditional on tools)
    system_message = SystemMessage(
        content=AgenticPromptTemplates.get_system_prompt(tools_enabled=tools_enabled)
    )

    # Create the initial user message with the question and context
    user_content = AgenticPromptTemplates.format_initial_prompt(
        question=state["question"],
        context=state["context"],
        yes_means=state["yes_means"],
        no_means=state["no_means"],
        sample_date=state["sample_date"],
        market_price=state["market_price"],
        tools_enabled=tools_enabled,
    )
    user_message = HumanMessage(content=user_content)

    # Don't set iterations here - it's initialized to 0 in create_initial_state
    # and will be incremented in agent_node when the first LLM call happens
    return {
        "messages": [system_message, user_message],
    }


def agent_node(state: AgentState, model: Any) -> dict[str, Any]:
    """
    Agent node that processes the current state and decides next action.

    The model can either:
    1. Call the web_search tool to gather more information
    2. Provide a final prediction

    Args:
        state: Current agent state
        model: The LangGraph model with tools bound

    Returns:
        State update with new messages
    """
    # Invoke the model with current messages
    response = model.invoke(state["messages"])

    # Increment iteration count
    new_iterations = state["iterations"] + 1

    return {
        "messages": [response],
        "iterations": new_iterations,
    }


def forecast_node(state: AgentState, model: Any) -> dict[str, Any]:
    """
    Final forecast node that extracts the prediction from agent's reasoning.

    This node prompts the model to provide a structured final prediction
    if one hasn't been clearly stated yet.

    Args:
        state: Current agent state
        model: The LangGraph model (without tools, for final output)

    Returns:
        State update with prediction, confidence, and reasoning
    """
    # Check if last message already has a clear prediction
    last_message = state["messages"][-1] if state["messages"] else None

    if last_message and isinstance(last_message, AIMessage):
        content = _get_text_content(last_message.content)

        # Try to extract prediction from existing content
        prediction, confidence, reasoning = _extract_prediction(content)

        if prediction is not None:
            return {
                "messages": [],  # Explicit empty for consistency
                "prediction": prediction,
                "confidence": confidence,
                "reasoning": reasoning,
            }

    # If no clear prediction, ask for structured output
    final_prompt = HumanMessage(
        content=AgenticPromptTemplates.FINAL_PREDICTION_PROMPT
    )

    # Build messages list, handling any pending tool calls
    # Anthropic API requires tool_use blocks to be followed by tool_result blocks
    messages = list(state["messages"])
    added_messages = []  # Track messages we add for the state update

    # Check if the last AIMessage has unanswered tool calls
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            # Add placeholder tool results for pending tool calls
            for tool_call in last_msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_id = tool_call.get("id", "")
                    tool_name = tool_call.get("name", "unknown")
                else:
                    tool_id = getattr(tool_call, "id", "")
                    tool_name = getattr(tool_call, "name", "unknown")

                placeholder_msg = ToolMessage(
                    content="[Tool not executed - proceeding to final prediction. Please provide your prediction based on the information gathered so far.]",
                    name=tool_name,
                    tool_call_id=tool_id,
                )
                messages.append(placeholder_msg)
                added_messages.append(placeholder_msg)

    messages.append(final_prompt)
    added_messages.append(final_prompt)
    response = model.invoke(messages)
    added_messages.append(response)

    # Extract prediction from response (handle None content or list content)
    content = _get_text_content(response.content)
    prediction, confidence, reasoning = _extract_prediction(content)

    # Default to uncertain if extraction fails
    if prediction is None:
        prediction = "unknown"
        confidence = 50.0
        reasoning = content if content else "Unable to make prediction"

    return {
        "messages": added_messages,
        "prediction": prediction,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def _extract_prediction(content: str | None) -> tuple[str | None, float | None, str | None]:
    """
    Extract prediction, confidence, and reasoning from model output.

    Looks for structured output with XML-style tags:
    - <answer>yes|no</answer>
    - <confidence>0-100</confidence>
    - <think>...</think> or <reasoning>...</reasoning>

    Args:
        content: The model's output text

    Returns:
        Tuple of (prediction, confidence, reasoning)
    """
    if not content:
        return None, None, None

    prediction = None
    confidence = None
    reasoning = None

    # Extract answer
    answer_match = re.search(r"<answer>\s*(yes|no)\s*</answer>", content, re.IGNORECASE)
    if answer_match:
        prediction = answer_match.group(1).lower()

    # Extract confidence
    confidence_match = re.search(
        r"<confidence>\s*(\d+(?:\.\d+)?)\s*</confidence>", content, re.IGNORECASE
    )
    if confidence_match:
        confidence = float(confidence_match.group(1))
        # Normalize to 0-1 range
        if confidence > 1.0:
            confidence /= 100.0
        # Clamp to valid range
        confidence = max(0.0, min(1.0, confidence))

        # Convert to P(YES) - if model says NO with high confidence, invert
        # This handles the case where model outputs "confidence in prediction"
        # instead of "P(YES)" directly
        if prediction == "no" and confidence > 0.5:
            confidence = 1.0 - confidence

    # Extract reasoning (try multiple tag formats)
    reasoning_match = re.search(
        r"<(?:think|reasoning)>(.*?)</(?:think|reasoning)>",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # Use content before answer tag as reasoning
        if answer_match:
            reasoning = content[: answer_match.start()].strip()
        else:
            reasoning = content.strip()

    return prediction, confidence, reasoning


def tools_node(state: AgentState, tools: Sequence[BaseTool]) -> dict[str, Any]:
    """
    Custom tools node that executes tool calls and tracks search queries/results.

    This replaces LangGraph's ToolNode to properly populate the search_queries
    and search_results state fields for web_search tool calls.

    Args:
        state: Current agent state
        tools: Available tools (including web_search)

    Returns:
        State update with tool messages and search tracking
    """
    # Build tool lookup
    tool_map = {tool.name: tool for tool in tools}

    # Get the last message which should have tool calls
    last_message = state["messages"][-1] if state["messages"] else None

    if not last_message or not isinstance(last_message, AIMessage):
        error_msg = "tools_node called without AIMessage - routing error"
        logger.error(error_msg)
        raise ValueError(error_msg)

    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        error_msg = "tools_node called without tool_calls - routing error"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Execute each tool call
    tool_messages = []
    search_queries = []
    search_results = []

    for tool_call in tool_calls:
        # Safely extract tool call fields (handle different formats)
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")
        else:
            # Handle object-style tool calls (some LangChain versions)
            tool_name = getattr(tool_call, "name", "")
            tool_args = getattr(tool_call, "args", {})
            tool_id = getattr(tool_call, "id", "")

        # Ensure tool_args is a dict
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name not in tool_map:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            tool_messages.append(ToolMessage(
                content=error_msg,
                name=tool_name,
                tool_call_id=tool_id,
            ))
            # Track failed web_search attempts for benchmarking
            if tool_name == "web_search":
                query = tool_args.get("query", "")
                if query:
                    search_queries.append(query)
                search_results.append(f"ERROR: {error_msg}")
            continue

        try:
            # Execute the tool
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)

            # Track web_search specifically
            if tool_name == "web_search":
                query = tool_args.get("query", "")
                if query:
                    search_queries.append(query)
                if result:
                    search_results.append(str(result))

            # Create tool message
            tool_messages.append(ToolMessage(
                content=str(result) if result else "No result returned",
                name=tool_name,
                tool_call_id=tool_id,
            ))

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            tool_messages.append(ToolMessage(
                content=error_msg,
                name=tool_name,
                tool_call_id=tool_id,
            ))
            # Track failed web_search executions for benchmarking
            if tool_name == "web_search":
                query = tool_args.get("query", "")
                if query:
                    search_queries.append(query)
                search_results.append(f"ERROR: {error_msg}")

    return {
        "messages": tool_messages,
        "search_queries": search_queries,
        "search_results": search_results,
    }


def should_continue(state: AgentState) -> str:
    """
    Determine the next node based on the agent's last action.

    Returns:
        - "tools": If the agent called a tool
        - "forecast": If the agent is ready to make a prediction
    """
    # Check last message for tool calls
    if state["messages"]:
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            # Check for tool calls - continue to tools node
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Check for final prediction in content
            content = _get_text_content(last_message.content)
            if "<answer>" in content.lower():
                return "forecast"

    # Default to forecast if no tool calls
    return "forecast"
