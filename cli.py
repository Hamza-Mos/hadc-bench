"""
CLI entry point for the agentic forecasting benchmark.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import (
    AgenticAPIConfig,
    AGENTIC_MODEL_ROUTES,
    DEFAULT_BENCHMARK_DATASET,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MODELS,
    VALID_CHECKPOINTS,
)
from models.registry import list_available_models, get_model_info
from runners.runner import AgenticBenchmarkRunner, AgenticRunConfig

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    # Note: web_search logger is silenced separately when verbose mode is on (see cmd_run)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hadc-bench",
        description="Agentic Forecasting Benchmark for Kalshi Prediction Markets",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the agentic benchmark")
    run_parser.add_argument(
        "--model", "--models",
        dest="models",
        nargs="+",
        default=None,  # None = use all models from config
        help="Model(s) to benchmark (default: all models)",
    )
    run_parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default=DEFAULT_BENCHMARK_DATASET,
        help=f"Path to benchmark dataset (default: {DEFAULT_BENCHMARK_DATASET})",
    )
    run_parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,  # None = all 5 checkpoints
        help="Checkpoint names to evaluate (default: all 5)",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum agent iterations (default: 100, effectively unlimited)",
    )
    run_parser.add_argument(
        "--max-search-results",
        type=int,
        default=5,
        help="Maximum search results per query (default: 5)",
    )
    run_parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to include (default: all)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="agentic_results",
        help="Output directory for results (default: agentic_results)",
    )
    run_parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    run_parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Don't save detailed trace data",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verbose output (default: enabled, use --no-verbose to disable)",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug logging",
    )
    run_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path",
    )
    run_parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable web search tool (for prior belief measurement)",
    )

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")

    # Check keys command
    check_parser = subparsers.add_parser("check-keys", help="Check API key availability")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Run the benchmark."""
    setup_logging(debug=args.debug, log_file=args.log_file)

    # Use all models if none specified
    models = args.models if args.models else DEFAULT_MODELS

    # Use all checkpoints if none specified
    checkpoints = args.checkpoints if args.checkpoints else DEFAULT_CHECKPOINTS

    # Validate models
    for model in models:
        if model not in AGENTIC_MODEL_ROUTES:
            print(f"Error: Unknown model '{model}'")
            print(f"Available models: {', '.join(sorted(AGENTIC_MODEL_ROUTES.keys()))}")
            return 1

    # Validate checkpoints
    for cp in checkpoints:
        if cp not in VALID_CHECKPOINTS:
            print(f"Error: Unknown checkpoint '{cp}'")
            print(f"Available checkpoints: {', '.join(sorted(VALID_CHECKPOINTS))}")
            return 1

    # Check API keys
    api_config = AgenticAPIConfig()

    # Check required keys
    providers_needed = set()
    for model in models:
        provider, _ = AGENTIC_MODEL_ROUTES[model]
        providers_needed.add(provider)

    missing_keys = []
    for provider in providers_needed:
        if not api_config.validate(provider):
            missing_keys.append(provider)

    if missing_keys:
        print(f"Error: Missing API keys for: {', '.join(missing_keys)}")
        print("Set the appropriate environment variables:")
        key_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "grok": "XAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        for provider in missing_keys:
            print(f"  export {key_vars.get(provider, provider.upper() + '_API_KEY')}=...")
        return 1

    # Check SerpAPI key / no-tools mode
    if hasattr(args, 'no_tools') and args.no_tools:
        print("Web search: disabled (--no-tools mode)")
    elif not api_config.serpapi_api_key:
        print("Warning: SERPAPI_API_KEY not set. Web search will not work.")
        print("Set: export SERPAPI_API_KEY=...")
    else:
        print("Web search: enabled (SerpAPI)")

    # Create run config
    run_config = AgenticRunConfig(
        model_ids=models,
        benchmark_dataset_path=args.benchmark_dataset,
        checkpoints=checkpoints,
        categories=args.categories,
        max_iterations=args.max_iterations,
        max_search_results=args.max_search_results,
        save_traces=not args.no_traces,
        parallelism=args.parallelism,
        tools_enabled=not args.no_tools,
    )

    # Determine if verbose is enabled
    verbose_enabled = args.verbose

    # When verbose is on, silence web_search logs to avoid interleaving
    if verbose_enabled:
        logging.getLogger("kalshi_agentic.tools.web_search").setLevel(logging.WARNING)

    # Progress callback (only show when not verbose)
    def progress(message: str, current: int, total: int) -> None:
        if not verbose_enabled:
            pct = (current / total * 100) if total > 0 else 0
            print(f"\r{message}: {current}/{total} ({pct:.1f}%)", end="", flush=True)
            if current == total:
                print()

    def format_agent_trajectory(prediction) -> str:
        """Format the agent trajectory for verbose output."""
        if not prediction.raw_messages:
            return "(no trajectory recorded)"

        lines = []
        step = 0

        for msg in prediction.raw_messages:
            msg_type = msg.get("type", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content else ""

            if msg_type == "AIMessage" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    step += 1
                    tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else "unknown"
                    tool_args = tc.get("args", {}) if isinstance(tc, dict) else {}

                    if tool_name == "web_search" and isinstance(tool_args, dict):
                        query = tool_args.get("query", "(no query)")
                        lines.append(f"  [{step}] TOOL CALL: web_search")
                        lines.append(f"      Query: \"{query}\"")
                    else:
                        lines.append(f"  [{step}] TOOL CALL: {tool_name}")
                        lines.append(f"      Args: {tool_args}")

            elif msg_type == "ToolMessage":
                step += 1
                tool_name = msg.get("name", "unknown")
                if len(content) > 500:
                    result_preview = content[:500].rsplit(' ', 1)[0] + "..."
                else:
                    result_preview = content
                lines.append(f"  [{step}] TOOL RESULT: {tool_name}")
                for line in result_preview.split('\n')[:10]:  # Limit lines
                    lines.append(f"      {line}")

        return "\n".join(lines) if lines else "(no tool calls made)"

    # Verbose callback with rich formatting
    def verbose_callback(sample, prediction, current: int, total: int) -> None:
        if not verbose_enabled:
            return

        # Get market data
        market_price = sample.market_price
        actual_outcome = sample.outcome

        # Get volume and open interest from price_data (not market, which may be None)
        if sample.price_data:
            volume = sample.price_data.volume_contracts or 0
            open_interest = sample.price_data.open_interest_contracts or 0
        else:
            volume = 0
            open_interest = 0

        # Model outputs
        model_conf = prediction.confidence_normalized
        model_error = abs(model_conf - actual_outcome)
        market_error = abs(market_price - actual_outcome)
        error_diff = market_error - model_error

        # Determine result status
        if abs(error_diff) < 1e-6:
            result_status = "TIE"
            result_symbol = "="
        elif error_diff > 0:
            result_status = "MODEL BEATS MARKET"
            result_symbol = "✓"
        else:
            result_status = "MARKET BEATS MODEL"
            result_symbol = "✗"

        # Build output
        lines = []
        lines.append("")
        lines.append("═" * 80)
        # Show checkpoint info in header for clarity
        checkpoint_name = sample.days_before_close if sample.days_before_close else "unknown"
        lines.append(f"SAMPLE {current}/{total}: {sample.market_ticker} @ {checkpoint_name}")
        lines.append("═" * 80)
        lines.append("")

        # Question section
        lines.append("QUESTION:")
        lines.append(sample.question)
        lines.append("")

        # Context
        if sample.full_context:
            if len(sample.full_context) > 300:
                context_preview = sample.full_context[:300].rsplit(' ', 1)[0] + "..."
            else:
                context_preview = sample.full_context
            lines.append("CONTEXT:")
            lines.append(context_preview)
            lines.append("")

        # Yes/No meanings
        if sample.yes_means:
            lines.append(f"YES means: {sample.yes_means}")
        if sample.no_means and sample.no_means != sample.yes_means:
            lines.append(f"NO means: {sample.no_means}")
        lines.append("")

        # Market data section
        lines.append("─" * 80)
        lines.append("MARKET DATA:")
        lines.append("─" * 80)
        lines.append(f"Sample Date: {sample.sample_date.strftime('%Y-%m-%d')}")
        if sample.close_date:
            lines.append(f"Close Date: {sample.close_date.strftime('%Y-%m-%d')}")
        lines.append(f"Market Price: {market_price:.2f} ({market_price*100:.0f}% implied probability)")
        if volume > 0 or open_interest > 0:
            lines.append(f"Volume: {volume:,} | Open Interest: {open_interest:,}")
        lines.append(f"Actual Outcome: {'YES' if actual_outcome == 1.0 else 'NO'}")
        lines.append("")

        # Agent trajectory section
        n_searches = len(prediction.search_queries)
        lines.append("═" * 80)
        lines.append(f"AGENT TRAJECTORY ({prediction.iterations} iterations, {n_searches} web searches)")
        lines.append("═" * 80)
        lines.append("")

        # Tool calls and results (includes all web_search queries and results)
        lines.append("─" * 80)
        lines.append("TOOL CALLS & RESULTS:")
        lines.append("─" * 80)
        lines.append(format_agent_trajectory(prediction))
        lines.append("")

        # Model response section
        lines.append("═" * 80)
        lines.append("MODEL RESPONSE")
        lines.append("═" * 80)
        lines.append("")

        # Reasoning
        if prediction.reasoning:
            if len(prediction.reasoning) > 800:
                reasoning_preview = prediction.reasoning[:800].rsplit(' ', 1)[0] + "..."
            else:
                reasoning_preview = prediction.reasoning
            lines.append("REASONING:")
            lines.append("─" * 80)
            lines.append(reasoning_preview)
            lines.append("")

        # Parsed output
        lines.append("─" * 80)
        lines.append("PARSED OUTPUT:")
        lines.append("─" * 80)
        lines.append(f"Prediction: {prediction.prediction.upper()}")
        lines.append(f"P(YES): {model_conf:.2f} ({model_conf*100:.0f}%)")
        lines.append(f"Parse Success: {prediction.parse_success}")
        lines.append(f"Model ID: {prediction.model_id}")
        lines.append("")

        # Evaluation section
        lines.append("═" * 80)
        lines.append("EVALUATION")
        lines.append("═" * 80)
        lines.append(f"Model Error:  {model_error:.4f}")
        lines.append(f"Market Error: {market_error:.4f}")
        diff_sign = "+" if error_diff >= 0 else ""
        diff_label = "(model better)" if error_diff > 0 else "(market better)" if error_diff < 0 else "(tie)"
        lines.append(f"Difference:   {diff_sign}{error_diff:.4f} {diff_label}")
        lines.append("")
        lines.append(f">>> {result_symbol} {result_status} <<<")
        lines.append("═" * 80)

        print("\n".join(lines), flush=True)

    run_config.progress_callback = progress
    run_config.verbose_callback = verbose_callback

    # Run benchmark
    print(f"Running agentic benchmark with models: {', '.join(models)}")
    print(f"Benchmark dataset: {args.benchmark_dataset}")
    print(f"Checkpoints: {checkpoints}")
    if args.categories:
        print(f"Categories: {args.categories}")
    else:
        print("Categories: all")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Output directory: {args.output}")
    print()

    runner = AgenticBenchmarkRunner(
        output_dir=args.output,
        api_config=api_config,
    )

    try:
        results = runner.run(run_config)
    except Exception as e:
        logger.exception("Benchmark failed")
        print(f"\nError: {e}")
        return 1

    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model_id, model_results in results.items():
        print(f"\n{model_id}:")
        print(f"  Run ID: {model_results['run_id']}")
        print(f"  Samples: {model_results['n_predictions']}")

        for checkpoint, metrics in model_results.get("metrics_by_checkpoint", {}).items():
            print(f"\n  {checkpoint} ({metrics.n_samples} samples):")
            print(f"    Accuracy: {metrics.accuracy:.1%}")
            print(f"    Brier Score: {metrics.brier_score:.4f}")
            if metrics.brier_skill_score is not None:
                print(f"    Brier Skill Score: {metrics.brier_skill_score:.4f}")
            print(f"    ECE: {metrics.ece:.4f}")
            print(f"    Parse Success: {metrics.parse_success_rate:.1%}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output}")

    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    """List available models."""
    print("Available Agentic Models:")
    print("=" * 60)

    for model_name in list_available_models():
        info = get_model_info(model_name)
        print(f"\n{model_name}:")
        print(f"  Provider: {info['provider']}")
        print(f"  Model ID: {info['model_id']}")
        print(f"  Temperature: {info['temperature']}")

    print("\n" + "=" * 60)
    print("\nTo run with a model:")
    print("  python -m kalshi_agentic.cli run --model <model-name>")

    return 0


def cmd_check_keys(args: argparse.Namespace) -> int:
    """Check API key availability."""
    api_config = AgenticAPIConfig()

    print("API Key Status:")
    print("=" * 60)

    keys = [
        ("ANTHROPIC_API_KEY", api_config.anthropic_api_key, "anthropic"),
        ("OPENAI_API_KEY", api_config.openai_api_key, "openai"),
        ("GOOGLE_API_KEY", api_config.google_api_key, "gemini"),
        ("XAI_API_KEY", api_config.xai_api_key, "grok"),
        ("OPENROUTER_API_KEY", api_config.openrouter_api_key, "openrouter"),
        ("SERPAPI_API_KEY", api_config.serpapi_api_key, "serpapi"),
    ]

    all_ok = True
    for env_var, value, provider in keys:
        status = "OK" if value else "MISSING"
        if not value:
            all_ok = False

        # Find models using this provider
        models = [
            name for name, (prov, _) in AGENTIC_MODEL_ROUTES.items()
            if prov == provider
        ]
        models_str = f" (models: {', '.join(models)})" if models else ""

        print(f"  {env_var}: {status}{models_str}")

    print()
    if all_ok:
        print("All API keys configured!")
    else:
        print("Some API keys are missing. Set them as environment variables.")

    return 0 if all_ok else 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "check-keys":
        return cmd_check_keys(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
