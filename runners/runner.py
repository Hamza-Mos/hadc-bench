"""
Agentic benchmark runner for forecasting with LangGraph agents.
"""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# Import from kalshi_benchmark for reusable components
from data_loader.temporal_sampler import TemporalSample
from evaluation.metrics import BenchmarkMetrics

# Default timeout for agent execution (in seconds)
DEFAULT_AGENT_TIMEOUT = 600  # 10 minutes per sample (thinking models need more time)

from config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SEARCH_RESULTS,
    AgenticAPIConfig,
    AGENTIC_MODEL_ROUTES,
    CHECKPOINT_ORDER,
    DEFAULT_BENCHMARK_DATASET,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODELS,
    DEFAULT_RETRY_DELAY,
    VALID_CHECKPOINTS,
)
from data_loader.benchmark_loader import BenchmarkDatasetLoader
from models.registry import get_agentic_model
from tools.web_search import create_search_tool
from graph.builder import build_forecast_agent
from graph.state import create_initial_state

logger = logging.getLogger(__name__)


@dataclass
class AgenticRunConfig:
    """Configuration for a single agentic benchmark run."""

    model_ids: list[str] = field(default_factory=lambda: DEFAULT_MODELS)
    checkpoints: list[str] = field(default_factory=lambda: DEFAULT_CHECKPOINTS)
    benchmark_dataset_path: str = DEFAULT_BENCHMARK_DATASET
    categories: Optional[list[str]] = None  # None = all categories
    max_iterations: int = DEFAULT_MAX_ITERATIONS  # Effectively unlimited - agent runs until it provides an answer
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS
    agent_timeout: int = DEFAULT_AGENT_TIMEOUT  # Timeout per sample in seconds
    max_retries: int = DEFAULT_MAX_RETRIES  # Number of retries per sample on failure
    retry_delay: float = DEFAULT_RETRY_DELAY  # Delay between retries in seconds
    save_traces: bool = True
    parallelism: int = 1
    tools_enabled: bool = True  # Set to False for prior belief runs
    progress_callback: Optional[Callable[[str, int, int], None]] = None
    verbose_callback: Optional[Callable] = None


@dataclass
class AgenticPrediction:
    """Result of an agentic prediction."""

    sample: TemporalSample
    prediction: str  # "yes", "no", or "unknown"
    confidence: float  # 0-100 (or 0-1 normalized)
    reasoning: str
    model_id: str
    search_queries: list[str]
    search_results: list[str]
    iterations: int
    raw_messages: list[dict]
    parse_success: bool = True

    @property
    def confidence_normalized(self) -> float:
        """Return confidence as probability P(YES) in 0-1 range.

        Note: Inversion for NO predictions is handled during parsing in
        _extract_prediction(), so confidence already represents P(YES).
        """
        conf = self.confidence
        if conf is None:
            conf = 0.5  # Default to maximum uncertainty
        # Normalize to 0-1 range if needed (shouldn't be necessary after parsing fix)
        if conf > 1.0:
            conf = conf / 100.0
        # Clamp to valid probability range
        return max(0.0, min(1.0, conf))

    @classmethod
    def create_error(
        cls,
        sample: TemporalSample,
        model_id: str,
        error_msg: str,
    ) -> "AgenticPrediction":
        """Create an error prediction with default values.

        Used when agent execution fails (timeout, exception, etc.).
        """
        return cls(
            sample=sample,
            prediction="unknown",
            confidence=50.0,
            reasoning=error_msg,
            model_id=model_id,
            search_queries=[],
            search_results=[],
            iterations=0,
            raw_messages=[],
            parse_success=False,
        )


class AgenticResultStorage:
    """Storage for agentic benchmark results.

    Output format matches kalshi_benchmark's RunStorage:
    - config.json: Run configuration
    - traces.json: Full prediction logs with agent trajectories
    - summary.json: Aggregated metrics
    - summary.md: Human-readable summary
    """

    def __init__(
        self,
        run_id: str,
        output_dir: Path,
        save_traces: bool = True,
        tools_enabled: bool = True,
    ):
        self.run_id = run_id
        self.output_dir = output_dir / run_id
        self.save_traces = save_traces
        self.tools_enabled = tools_enabled

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config: dict = {}
        self.traces: list[dict] = []
        self.metrics: list[dict] = []
        self.started_at = datetime.now().isoformat()
        self.completed_at: Optional[str] = None
        self.status = "running"

    def set_config(self, config: dict) -> None:
        """Save run configuration."""
        self.config = config
        self._save_config()

    def _save_config(self) -> None:
        """Save config file."""
        config_data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "status": self.status,
            **self.config,
        }
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

    def add_prediction(
        self,
        prediction: AgenticPrediction,
        actual_outcome: float,
    ) -> None:
        """Add a prediction result to traces.

        Matches kalshi_benchmark trace format plus agentic-specific fields.
        """
        sample = prediction.sample
        model_confidence = prediction.confidence_normalized
        market_price = sample.market_price

        # Compute scores
        brier_score = (model_confidence - actual_outcome) ** 2
        market_brier_score = (market_price - actual_outcome) ** 2
        model_error = abs(model_confidence - actual_outcome)
        market_error = abs(market_price - actual_outcome)

        # Determine result status (win/loss/tie)
        diff = market_error - model_error
        if abs(diff) < 1e-6:
            result_status = "tie"
            beats_market = False
        elif diff > 0:
            result_status = "win"
            beats_market = True
        else:
            result_status = "loss"
            beats_market = False

        # Extract volume and open_interest from market if available
        volume = sample.market.total_volume_contracts if sample.market else 0
        open_interest = sample.market.open_interest_contracts if sample.market else 0

        # Build trace record matching kalshi_benchmark format
        record = {
            # Identification
            "model_id": prediction.model_id,
            "strategy": "agentic",
            "market_ticker": sample.market_ticker,
            "temporal_days": sample.days_before_close,
            "sample_date": sample.sample_date.isoformat(),
            "close_date": sample.close_date.isoformat() if sample.close_date else "",
            "timestamp": datetime.now().isoformat(),

            # Question and context
            "question": sample.question,
            "full_context": sample.full_context,
            "yes_means": sample.yes_means,
            "no_means": sample.no_means,

            # Market data
            "market_price": market_price,
            "volume": volume,
            "open_interest": open_interest,
            "actual_outcome": actual_outcome,
            "outcome_label": "YES" if actual_outcome == 1.0 else "NO",

            # Model output
            "model_prediction": prediction.prediction,
            "model_confidence": model_confidence,
            "model_reasoning": prediction.reasoning,
            "parse_success": prediction.parse_success,
            "raw_response": self._format_raw_response(prediction.raw_messages),

            # Scores
            "brier_score": brier_score,
            "market_brier_score": market_brier_score,
            "model_error": model_error,
            "market_error": market_error,
            "beats_market": beats_market,
            "result_status": result_status,

            # Agentic-specific fields
            "iterations": prediction.iterations,
            "search_queries": prediction.search_queries,
            "agent_trajectory": self._extract_trajectory(prediction.raw_messages),
            "tools_enabled": self.tools_enabled,
        }

        # Include full search results only if save_traces enabled
        if self.save_traces:
            record["search_results"] = prediction.search_results

        self.traces.append(record)

    def _format_raw_response(self, raw_messages: list[dict]) -> str:
        """Format raw messages into a single response string."""
        if not raw_messages:
            return ""
        # Get the last message content as the "raw response"
        for msg in reversed(raw_messages):
            content = msg.get("content", "")
            if content and isinstance(content, str):
                return content
        return ""

    def _extract_trajectory(self, raw_messages: list[dict]) -> list[dict]:
        """Extract structured agent trajectory from raw messages.

        Converts LangGraph message format to a structured trajectory format:
        - step: Sequential step number
        - node: Which graph node (research, agent, tools, forecast)
        - action: What happened (tool_call, tool_result, reasoning, etc.)
        - Additional fields based on action type

        Special handling for web_search:
        - Uses "search_query" field instead of generic "tool_input"
        - Larger content limit (1000 chars) for search results
        """
        trajectory = []
        step = 0

        for msg in raw_messages:
            step += 1
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
            # Ensure content is a string (should already be after serialization fix)
            if not isinstance(content, str):
                content = str(content) if content else ""

            # Determine node and action based on message type
            if msg_type == "HumanMessage":
                trajectory.append({
                    "step": step,
                    "node": "input",
                    "action": "user_input",
                    "content": content[:500] if content else "",
                })
            elif msg_type == "AIMessage":
                # Check if it's a tool call or reasoning
                if "tool_calls" in msg:
                    for tool_call in msg.get("tool_calls", []):
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})

                        entry = {
                            "step": step,
                            "node": "agent",
                            "action": "tool_call",
                            "tool": tool_name,
                        }

                        # Special handling for web_search: use search_query field
                        if tool_name == "web_search":
                            entry["search_query"] = tool_args.get("query", "")
                        else:
                            entry["tool_input"] = tool_args

                        trajectory.append(entry)
                else:
                    trajectory.append({
                        "step": step,
                        "node": "agent",
                        "action": "reasoning",
                        "content": content[:500] if content else "",
                    })
            elif msg_type == "ToolMessage":
                tool_name = msg.get("name", "unknown")
                # Use larger content limit for web_search results (1000 chars)
                content_limit = 1000 if tool_name == "web_search" else 500
                trajectory.append({
                    "step": step,
                    "node": "tools",
                    "action": "tool_result",
                    "tool": tool_name,
                    "content": content[:content_limit] if content else "",
                })
            else:
                # Generic message
                trajectory.append({
                    "step": step,
                    "node": "unknown",
                    "action": msg_type.lower() if msg_type else "unknown",
                    "content": content[:500] if isinstance(content, str) and content else "",
                })

        return trajectory

    def add_metrics(
        self,
        model_id: str,
        checkpoint: Optional[str],
        metrics: BenchmarkMetrics,
    ) -> None:
        """Add computed metrics."""
        self.metrics.append({
            "model_id": model_id,
            "strategy": "agentic",
            "checkpoint": checkpoint,
            **metrics.to_dict(),
        })

    def complete(self, status: str = "completed") -> None:
        """Finalize and save all results.

        Saves:
        - traces.json: Full prediction logs with agent trajectories
        - summary.json: Aggregated metrics and run info
        - summary.md: Human-readable summary
        """
        self.completed_at = datetime.now().isoformat()
        self.status = status
        self._save_all()

    def _save_all(self) -> None:
        """Save all data files."""
        # Update config with final status
        config_data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            **self.config,
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        # Save traces (full prediction logs)
        if self.save_traces and self.traces:
            with open(self.output_dir / "traces.json", "w") as f:
                json.dump(self.traces, f, indent=2, default=str)

        # Save summary JSON
        self._save_summary_json()

        # Save summary markdown
        self._save_summary_md()

        logger.info(f"Results saved to {self.output_dir}")

    def _save_summary_json(self) -> None:
        """Save summary.json with key metrics."""
        total_predictions = len(self.traces)
        beats_market_count = sum(1 for t in self.traces if t.get("beats_market"))
        tie_count = sum(1 for t in self.traces if t.get("result_status") == "tie")
        parse_success_count = sum(1 for t in self.traces if t.get("parse_success"))

        summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "total_predictions": total_predictions,
            "parse_success_rate": parse_success_count / total_predictions if total_predictions else 0,
            "beats_market_rate": beats_market_count / total_predictions if total_predictions else 0,
            "tie_rate": tie_count / total_predictions if total_predictions else 0,
            "config": self.config,
            "metrics": self.metrics,
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _save_summary_md(self) -> None:
        """Save human-readable markdown summary."""
        total_predictions = len(self.traces)
        beats_market_count = sum(1 for t in self.traces if t.get("beats_market"))
        tie_count = sum(1 for t in self.traces if t.get("result_status") == "tie")

        lines = [
            f"# Benchmark Run: {self.run_id}",
            "",
            f"Started: {self.started_at}",
            f"Completed: {self.completed_at}",
            f"Status: {self.status}",
            "",
            "## Configuration",
            "",
        ]

        for key, value in self.config.items():
            lines.append(f"- {key}: {value}")

        lines.extend([
            "",
            "## Results",
            "",
            f"Total predictions: {total_predictions}",
        ])

        if total_predictions:
            lines.append(f"Beats market: {beats_market_count}/{total_predictions} ({beats_market_count/total_predictions:.1%})")
            lines.append(f"Ties: {tie_count}/{total_predictions} ({tie_count/total_predictions:.1%})")

        lines.append("")

        # Group metrics - show overall first, then by checkpoint
        overall_metrics = [m for m in self.metrics if m.get("checkpoint") is None]
        for m in overall_metrics:
            bss = m.get("brier_skill_score")
            bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
            beats = "YES" if bss and bss > 0 else "NO"

            lines.extend([
                f"### {m['model_id']} ({m['strategy']})",
                "",
                f"- Samples: {m['n_samples']}",
                f"- Brier Score: {m['brier_score']:.4f}",
                f"- Brier Skill Score: {bss_str} (beats market: {beats})",
                f"- ECE: {m['ece']:.4f}",
                f"- Accuracy: {m['accuracy']:.3f}",
                f"- F1: {m['f1']:.3f}",
                "",
            ])

        # Show checkpoint breakdown
        checkpoint_metrics = [m for m in self.metrics if m.get("checkpoint") is not None]
        if checkpoint_metrics:
            lines.append("### Checkpoint Breakdown")
            lines.append("")
            # Sort by checkpoint name
            def sort_key(m):
                cp = m.get("checkpoint", "")
                return CHECKPOINT_ORDER.index(cp) if cp in CHECKPOINT_ORDER else 999
            for m in sorted(checkpoint_metrics, key=sort_key):
                cp = m.get("checkpoint")
                bss = m.get("brier_skill_score")
                bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
                lines.append(f"- {cp}: Brier={m['brier_score']:.4f}, BSS={bss_str}, n={m['n_samples']}")
            lines.append("")

        with open(self.output_dir / "summary.md", "w") as f:
            f.write("\n".join(lines))


class AgenticBenchmarkRunner:
    """
    Orchestrates agentic forecasting benchmark runs.

    Manages:
    - Benchmark dataset loading
    - Agent execution with web search
    - Metrics computation
    - Result storage
    """

    def __init__(
        self,
        output_dir: str | Path = "results",
        api_config: Optional[AgenticAPIConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.api_config = api_config or AgenticAPIConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        config: AgenticRunConfig,
    ) -> dict:
        """
        Run the agentic benchmark.

        Args:
            config: Run configuration

        Returns:
            Dictionary with results and metrics

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If benchmark dataset doesn't exist
        """
        # Validate models
        for model_id in config.model_ids:
            if model_id not in AGENTIC_MODEL_ROUTES:
                available = ", ".join(sorted(AGENTIC_MODEL_ROUTES.keys()))
                raise ValueError(
                    f"Unknown model: {model_id}. Available: {available}"
                )

        # Validate benchmark dataset path exists
        dataset_path = Path(config.benchmark_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Benchmark dataset not found: {config.benchmark_dataset_path}"
            )

        # Validate checkpoints
        for cp in config.checkpoints:
            if cp not in VALID_CHECKPOINTS:
                raise ValueError(
                    f"Invalid checkpoint: {cp}. Valid: {sorted(VALID_CHECKPOINTS)}"
                )

        # Validate parallelism
        if config.parallelism < 1:
            raise ValueError(f"Parallelism must be >= 1, got {config.parallelism}")

        # Validate timeout
        if config.agent_timeout < 10:
            raise ValueError(f"Agent timeout must be >= 10 seconds, got {config.agent_timeout}")

        # Load samples from benchmark dataset
        logger.info(f"Loading benchmark dataset from {config.benchmark_dataset_path}...")
        loader = BenchmarkDatasetLoader(config.benchmark_dataset_path)

        samples = loader.load_samples(
            checkpoints=config.checkpoints,
            categories=config.categories,
        )
        logger.info(f"Loaded {len(samples)} samples")

        if not samples:
            raise ValueError("No samples loaded from benchmark dataset")

        # Log breakdown
        metadata = loader.metadata
        logger.info(f"Dataset version: {metadata.version}")
        logger.info(f"Checkpoints: {config.checkpoints}")
        if config.categories:
            logger.info(f"Categories: {config.categories}")
        else:
            logger.info(f"Categories: all ({metadata.categories})")

        # Run for each model
        all_results = {}
        for model_id in config.model_ids:
            logger.info(f"Running benchmark for {model_id}...")
            model_results = self._run_single_model(
                model_id=model_id,
                samples=samples,
                config=config,
            )
            all_results[model_id] = model_results

        return all_results

    def _run_single_model(
        self,
        model_id: str,
        samples: list[TemporalSample],
        config: AgenticRunConfig,
    ) -> dict:
        """Run benchmark for a single model."""
        # Create run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:4]
        suffix = "_no_tools" if not config.tools_enabled else ""
        run_id = f"run_{timestamp}_{short_id}_{model_id}{suffix}_after_fix"

        # Initialize storage
        storage = AgenticResultStorage(
            run_id=run_id,
            output_dir=self.output_dir,
            save_traces=config.save_traces,
            tools_enabled=config.tools_enabled,
        )

        # Save config (format matches kalshi_benchmark)
        storage.set_config({
            "model_ids": [model_id],
            "strategies": ["agentic"],
            "benchmark_dataset": config.benchmark_dataset_path,
            "checkpoints": config.checkpoints,
            "categories": config.categories,
            "max_iterations": config.max_iterations,
            "max_search_results": config.max_search_results,
            "parallelism": config.parallelism,
            "tools_enabled": config.tools_enabled,
            "n_samples": len(samples),
        })

        # Get model
        model = get_agentic_model(model_id, api_config=self.api_config)

        # Run predictions
        predictions = self._run_predictions(
            model=model,
            model_id=model_id,
            samples=samples,
            config=config,
            storage=storage,
        )

        # Compute overall metrics (checkpoint=None)
        if predictions:
            overall_metrics = self._compute_metrics(predictions)
            storage.add_metrics(model_id, None, overall_metrics)

        # Compute metrics by checkpoint
        metrics_by_checkpoint = {}
        for checkpoint in config.checkpoints:
            checkpoint_predictions = [
                p for p in predictions
                if p.sample.days_before_close == checkpoint
            ]
            if checkpoint_predictions:
                metrics = self._compute_metrics(checkpoint_predictions)
                metrics_by_checkpoint[checkpoint] = metrics
                storage.add_metrics(model_id, checkpoint, metrics)

        # Finalize storage
        storage.complete()

        return {
            "run_id": run_id,
            "n_samples": len(samples),
            "n_predictions": len(predictions),
            "metrics_by_checkpoint": metrics_by_checkpoint,
        }

    def _run_predictions(
        self,
        model,
        model_id: str,
        samples: list[TemporalSample],
        config: AgenticRunConfig,
        storage: AgenticResultStorage,
    ) -> list[AgenticPrediction]:
        """Run predictions on samples."""
        predictions = []
        progress_counter = {"current": 0}
        progress_lock = threading.Lock()
        storage_lock = threading.Lock()
        print_lock = threading.Lock()

        def process_sample(sample: TemporalSample) -> AgenticPrediction:
            try:
                prediction = self._run_single_prediction(
                    model=model,
                    model_id=model_id,
                    sample=sample,
                    config=config,
                )
            except Exception as e:
                logger.error(f"Error processing {sample.market_ticker}: {e}")
                prediction = AgenticPrediction.create_error(
                    sample=sample,
                    model_id=model_id,
                    error_msg=f"Error: {str(e)}",
                )

            # Update storage
            with storage_lock:
                storage.add_prediction(prediction, sample.outcome)

            # Update progress counter
            with progress_lock:
                progress_counter["current"] += 1
                current = progress_counter["current"]

            # Progress callback (outside lock, uses captured value)
            if config.progress_callback:
                config.progress_callback(
                    f"Processing {model_id}",
                    current,
                    len(samples),
                )

            # Verbose output (separate lock)
            if config.verbose_callback:
                with print_lock:
                    config.verbose_callback(
                        sample=sample,
                        prediction=prediction,
                        current=current,
                        total=len(samples),
                    )

            return prediction

        if config.parallelism > 1:
            with ThreadPoolExecutor(max_workers=config.parallelism) as executor:
                futures = {
                    executor.submit(process_sample, sample): sample
                    for sample in samples
                }
                for future in as_completed(futures):
                    predictions.append(future.result())
        else:
            for sample in samples:
                predictions.append(process_sample(sample))

        return predictions

    def _run_single_prediction(
        self,
        model,
        model_id: str,
        sample: TemporalSample,
        config: AgenticRunConfig,
    ) -> AgenticPrediction:
        """Run a single agentic prediction with timeout protection and retries.

        Implements retry logic similar to kalshi_benchmark: retries up to
        max_retries times on any failure (timeout, API errors, parse failures).
        Only fails if ALL retries fail.
        """
        last_error = None

        for attempt in range(config.max_retries):
            try:
                result = self._execute_agent_once(
                    model=model,
                    model_id=model_id,
                    sample=sample,
                    config=config,
                )

                # Check if we got a valid prediction
                prediction = result.get("prediction", "unknown")
                if prediction in ("yes", "no"):
                    # Success - return the prediction
                    return self._build_prediction_from_result(
                        result=result,
                        sample=sample,
                        model_id=model_id,
                    )

                # Parse failed but no exception - still retry
                last_error = f"Parse failed: prediction='{prediction}'"
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries} failed for "
                    f"{sample.market_ticker}: {last_error}"
                )

            except FuturesTimeoutError:
                last_error = f"Agent timed out after {config.agent_timeout} seconds"
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries} timed out for "
                    f"{sample.market_ticker}"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries} failed for "
                    f"{sample.market_ticker}: {e}"
                )

            # Wait before retry (unless this was the last attempt)
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay)

        # All retries failed
        logger.error(
            f"All {config.max_retries} attempts failed for {sample.market_ticker}: {last_error}"
        )
        return AgenticPrediction.create_error(
            sample=sample,
            model_id=model_id,
            error_msg=f"All {config.max_retries} retries failed. Last error: {last_error}",
        )

    def _execute_agent_once(
        self,
        model,
        model_id: str,
        sample: TemporalSample,
        config: AgenticRunConfig,
    ) -> dict:
        """Execute a single agent run (no retries). May raise exceptions."""
        # Create tools list based on config
        if config.tools_enabled:
            search_tool = create_search_tool(
                sample_date=sample.sample_date,
                api_key=self.api_config.serpapi_api_key,
                max_results=config.max_search_results,
            )
            tools = [search_tool]
        else:
            tools = []

        # Build the agent fresh for each attempt
        agent = build_forecast_agent(
            model=model,
            tools=tools,
            max_iterations=config.max_iterations,
        )

        # Create initial state
        # NOTE: market_price intentionally set to None to avoid data leakage.
        # Showing the model the market price causes it to anchor to that value,
        # making "beats market" evaluation meaningless.
        initial_state = create_initial_state(
            question=sample.question,
            context=sample.full_context,
            yes_means=sample.yes_means,
            no_means=sample.no_means,
            sample_date=sample.sample_date.strftime("%Y-%m-%d"),
            market_price=None,
            max_iterations=config.max_iterations,
            tools_enabled=config.tools_enabled,
        )

        # Run the agent with timeout protection
        def run_agent():
            return agent.invoke(initial_state)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_agent)
            result = future.result(timeout=config.agent_timeout)

        return result

    def _build_prediction_from_result(
        self,
        result: dict,
        sample: TemporalSample,
        model_id: str,
    ) -> AgenticPrediction:
        """Build AgenticPrediction from successful agent result."""
        prediction = result.get("prediction", "unknown")
        confidence = result.get("confidence", 50.0)
        reasoning = result.get("reasoning", "")
        iterations = result.get("iterations", 0)
        search_queries = result.get("search_queries", [])
        search_results = result.get("search_results", [])

        # Convert messages to serializable format
        raw_messages = []
        for msg in result.get("messages", []):
            try:
                # Handle content that may be a list of content blocks (Anthropic models)
                raw_content = msg.content if hasattr(msg, "content") else str(msg)
                if isinstance(raw_content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in raw_content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    raw_content = "\n".join(text_parts)
                msg_dict = {
                    "type": msg.__class__.__name__,
                    "content": raw_content,
                }
                # Capture tool calls if present (for AIMessage)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_list = []
                    for tc in msg.tool_calls:
                        # Handle both dict and object-style tool calls
                        if isinstance(tc, dict):
                            tool_calls_list.append({
                                "id": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {})
                            })
                        else:
                            tool_calls_list.append({
                                "id": getattr(tc, "id", ""),
                                "name": getattr(tc, "name", ""),
                                "args": getattr(tc, "args", {})
                            })
                    msg_dict["tool_calls"] = tool_calls_list
                # Capture tool name if present (for ToolMessage)
                if hasattr(msg, "name") and msg.name:
                    msg_dict["name"] = msg.name
                raw_messages.append(msg_dict)
            except Exception as e:
                # If message serialization fails, add a placeholder
                logger.warning(f"Failed to serialize message: {e}")
                raw_messages.append({"type": "unknown", "content": str(msg)})

        parse_success = prediction in ("yes", "no")

        return AgenticPrediction(
            sample=sample,
            prediction=prediction,
            confidence=confidence if confidence is not None else 50.0,
            reasoning=reasoning or "",
            model_id=model_id,
            search_queries=search_queries,
            search_results=search_results,
            iterations=iterations,
            raw_messages=raw_messages,
            parse_success=parse_success,
        )

    def _compute_metrics(
        self,
        predictions: list[AgenticPrediction],
    ) -> BenchmarkMetrics:
        """Compute benchmark metrics from predictions."""
        model_predictions = []
        market_predictions = []
        outcomes = []
        parse_successes = []

        for pred in predictions:
            model_predictions.append(pred.confidence_normalized)
            market_predictions.append(pred.sample.market_price)
            outcomes.append(pred.sample.outcome)
            parse_successes.append(pred.parse_success)

        return BenchmarkMetrics.compute(
            model_predictions=model_predictions,
            market_predictions=market_predictions,
            outcomes=outcomes,
            parse_successes=parse_successes,
        )
