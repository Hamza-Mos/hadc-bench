"""
Prompt templates for the agentic forecasting agent.
"""

SYSTEM_PROMPT_BASE = """You are an expert forecaster participating in a prediction market challenge.

Your task is to analyze prediction market questions and provide probability estimates.{tools_section}

## Your Approach

1. **Understand the Question**: Carefully read the prediction market question and understand what outcome you're predicting.

2. **{research_verb} Strategically**: Consider relevant factors:
   - Recent news and developments related to the topic
   - Historical data and precedents
   - Expert opinions and analysis
   - Statistical data if relevant

3. **Synthesize Information**: Combine your {knowledge_source} with your reasoning to form a probability estimate.

4. **Provide Your Prediction**: Give a clear yes/no prediction with a confidence level (0-100%).

## Important Guidelines

- Consider base rates and historical frequencies
- Account for uncertainty in your confidence level
- Be calibrated: a 70% confidence means you expect to be right about 70% of the time

## Output Format

When you're ready to make your final prediction, format your response as:

<think>
Your reasoning process and how you arrived at your conclusion.
</think>

<answer>yes</answer> or <answer>no</answer>

<confidence>0-100 representing your confidence that the answer is "yes"</confidence>

Be calibrated: if you're 70% confident, you should be correct 70% of the time.
Do not hedge or give non-committal answers. Make a clear prediction."""

TOOLS_SECTION = """

You have access to a web search tool that allows you to gather relevant information before making your prediction.

- The web search results are filtered by date to prevent future data leakage
- Focus on information that would genuinely inform the prediction"""

INITIAL_PROMPT_BASE = """## Prediction Market Question

**Question**: {question}

**Context**: {context}

**What YES means**: {yes_means}
**What NO means**: {no_means}

**Current Date**: {sample_date}
{market_price_section}

{instruction}

Provide your prediction in the specified format with <think>, <answer>, and <confidence> tags."""

FINAL_PREDICTION_PROMPT = """Based on your research and analysis, please provide your final prediction now.

Format your response as:

<think>
Summarize your key reasoning and the most important factors that influenced your prediction.
</think>

<answer>yes</answer> or <answer>no</answer>

<confidence>0-100 representing your confidence that the answer is "yes"</confidence>

Be calibrated and make a clear prediction."""


def get_system_prompt(tools_enabled: bool = True) -> str:
    """Get the system prompt, optionally including tool instructions."""
    return SYSTEM_PROMPT_BASE.format(
        tools_section=TOOLS_SECTION if tools_enabled else "",
        research_verb="Research" if tools_enabled else "Analyze",
        knowledge_source="research findings" if tools_enabled else "knowledge",
    )


def format_initial_prompt(
    question: str,
    context: str,
    yes_means: str,
    no_means: str,
    sample_date: str,
    market_price: float | None = None,
    tools_enabled: bool = True,
) -> str:
    """Format the initial prompt with market information."""
    market_price_section = ""
    if market_price is not None:
        market_price_section = f"\n**Current Market Price**: {market_price:.1%} implied probability of YES"

    if tools_enabled:
        instruction = "Please research this question and provide your prediction. You may use the web_search tool to gather relevant information before making your final prediction."
    else:
        instruction = "Please analyze this question and provide your prediction."

    return INITIAL_PROMPT_BASE.format(
        question=question,
        context=context,
        yes_means=yes_means,
        no_means=no_means,
        sample_date=sample_date,
        market_price_section=market_price_section,
        instruction=instruction,
    )


# Backwards compatibility - wrap functions in a class
class AgenticPromptTemplates:
    """Backwards-compatible wrapper for prompt template functions."""

    FINAL_PREDICTION_PROMPT = FINAL_PREDICTION_PROMPT

    @staticmethod
    def get_system_prompt(tools_enabled: bool = True) -> str:
        return get_system_prompt(tools_enabled)

    @staticmethod
    def format_initial_prompt(
        question: str,
        context: str,
        yes_means: str,
        no_means: str,
        sample_date: str,
        market_price: float | None = None,
        tools_enabled: bool = True,
    ) -> str:
        return format_initial_prompt(
            question=question,
            context=context,
            yes_means=yes_means,
            no_means=no_means,
            sample_date=sample_date,
            market_price=market_price,
            tools_enabled=tools_enabled,
        )
