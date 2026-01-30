"""
Loader for pre-computed benchmark datasets.

Loads samples from benchmark_dataset_v2.json format and converts
them to TemporalSample objects for use with the agentic runner.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from data_loader.temporal_sampler import TemporalSample
from data_loader.loader import PriceData

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDatasetMetadata:
    """Metadata from the benchmark dataset file."""

    version: int
    created_at: str
    categories: list[str]
    checkpoints: list[str]
    total_samples: int
    unique_markets: int
    by_category: dict[str, int]
    by_checkpoint: dict[str, int]


class BenchmarkDatasetLoader:
    """
    Loader for pre-computed benchmark datasets.

    Loads samples from benchmark_dataset_v2.json and converts them
    to TemporalSample format for use with the agentic benchmark runner.
    """

    def __init__(self, dataset_path: str | Path):
        """
        Initialize the loader.

        Args:
            dataset_path: Path to the benchmark dataset JSON file
        """
        self.dataset_path = Path(dataset_path)
        self._data: Optional[dict] = None
        self._metadata: Optional[BenchmarkDatasetMetadata] = None

    def _load_data(self) -> dict:
        """Load and cache the dataset."""
        if self._data is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            with open(self.dataset_path) as f:
                self._data = json.load(f)
        return self._data

    @property
    def metadata(self) -> BenchmarkDatasetMetadata:
        """Get dataset metadata."""
        if self._metadata is None:
            data = self._load_data()
            self._metadata = BenchmarkDatasetMetadata(
                version=data.get("version", 1),
                created_at=data.get("created_at", ""),
                categories=data.get("categories", []),
                checkpoints=data.get("checkpoints", []),
                total_samples=data.get("total_samples", 0),
                unique_markets=data.get("unique_markets", 0),
                by_category=data.get("by_category", {}),
                by_checkpoint=data.get("by_checkpoint", {}),
            )
        return self._metadata

    def load_samples(
        self,
        checkpoints: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> list[TemporalSample]:
        """
        Load samples from the benchmark dataset.

        Args:
            checkpoints: Optional list of checkpoint names to filter by
                        (e.g., ["pct_25", "pct_50"]). If None, loads all.
            categories: Optional list of category names to filter by
                       (e.g., ["Politics/Elections", "Financial"]). If None, loads all.

        Returns:
            List of TemporalSample objects
        """
        data = self._load_data()
        raw_samples = data.get("samples", [])

        samples = []
        # Track close_date by ticker for samples where close_time is null
        close_date_by_ticker: dict[str, datetime] = {}

        for raw in raw_samples:
            # Filter by checkpoint
            if checkpoints is not None:
                if raw.get("checkpoint_name") not in checkpoints:
                    continue

            # Filter by category
            if categories is not None:
                if raw.get("category") not in categories:
                    continue

            ticker = raw.get("ticker", "")
            prev_close_date = close_date_by_ticker.get(ticker)

            sample = self._convert_sample(raw, prev_close_date)
            if sample is not None:
                samples.append(sample)
                # Update the close_date tracking for this ticker
                close_date_by_ticker[ticker] = sample.close_date

        return samples

    def _convert_sample(
        self, raw: dict, prev_close_date: Optional[datetime] = None
    ) -> Optional[TemporalSample]:
        """
        Convert a raw sample dict to a TemporalSample object.

        Field mappings:
        - checkpoint_name → days_before_close (stored as string)
        - checkpoint_date → sample_date
        - close_time → close_date
        - price_at_checkpoint.price.close / 100 → market_price
        - result → outcome (yes=1.0, no=0.0)
        - title → question
        - rules_primary → full_context
        - yes_sub_title → yes_means
        - no_sub_title → no_means
        """
        try:
            # Parse dates
            checkpoint_date = raw.get("checkpoint_date", "")
            sample_date = datetime.strptime(checkpoint_date, "%Y-%m-%d")

            close_time = raw.get("close_time")
            # Handle ISO format with timezone
            # If close_time is null, use the previous close_date (same market)
            if close_time:
                # Remove microseconds and timezone for parsing
                close_time_clean = close_time.split(".")[0].replace("Z", "")
                close_date = datetime.fromisoformat(close_time_clean)
            elif prev_close_date:
                close_date = prev_close_date
            else:
                close_date = sample_date  # Last resort fallback

            # Extract market price from checkpoint data
            # Prefer close price, fall back to yes_ask close if close is None
            price_at_checkpoint = raw.get("price_at_checkpoint", {})
            price_data = price_at_checkpoint.get("price", {})
            yes_ask = price_at_checkpoint.get("yes_ask", {})

            price_close = price_data.get("close")
            if price_close is None:
                # Fall back to yes_ask close
                price_close = yes_ask.get("close", 50)

            # Ensure price_close is numeric
            try:
                price_close = float(price_close) if price_close is not None else 50.0
            except (ValueError, TypeError):
                price_close = 50.0

            market_price = price_close / 100.0  # Convert cents to probability

            # Convert result to outcome
            result = raw.get("result", "").lower()
            if result == "yes":
                outcome = 1.0
            elif result == "no":
                outcome = 0.0
            else:
                # Skip samples without clear outcome
                return None

            # Build PriceData object for compatibility
            # Note: We create a minimal PriceData since the full structure isn't needed
            yes_ask = price_at_checkpoint.get("yes_ask", {})
            yes_bid = price_at_checkpoint.get("yes_bid", {})

            # Safely extract numeric values
            def safe_int(val, default=0):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            pd = PriceData(
                date=checkpoint_date,
                timestamp=safe_int(price_at_checkpoint.get("end_period_ts")),
                open=price_data.get("open"),
                high=price_data.get("high"),
                low=price_data.get("low"),
                close=price_close,
                mean=price_data.get("mean"),
                volume_contracts=safe_int(price_at_checkpoint.get("volume")),
                open_interest_contracts=safe_int(price_at_checkpoint.get("open_interest")),
                yes_ask_close=yes_ask.get("close"),
                yes_bid_close=yes_bid.get("close"),
            )

            # Create TemporalSample
            # Use checkpoint_name as days_before_close (string stored in the field)
            # The runner will need to handle string checkpoint names
            return TemporalSample(
                market_ticker=raw.get("ticker", ""),
                market=None,  # We don't have the full Market object
                days_before_close=raw.get("checkpoint_name", "unknown"),  # Store checkpoint name
                sample_date=sample_date,
                close_date=close_date,
                price_data=pd,
                market_price=market_price,
                outcome=outcome,
                question=raw.get("title", ""),
                full_context=raw.get("rules_primary", ""),
                yes_means=raw.get("yes_sub_title", ""),
                no_means=raw.get("no_sub_title", ""),
            )
        except Exception as e:
            # Log and skip malformed samples
            ticker = raw.get("ticker", "unknown")
            checkpoint = raw.get("checkpoint_name", "unknown")
            logger.warning(f"Skipping malformed sample {ticker}@{checkpoint}: {e}")
            return None

    def get_available_checkpoints(self) -> list[str]:
        """Get list of available checkpoint names."""
        return self.metadata.checkpoints

    def get_available_categories(self) -> list[str]:
        """Get list of available category names."""
        return self.metadata.categories
