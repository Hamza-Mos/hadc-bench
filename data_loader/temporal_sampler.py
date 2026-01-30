"""
Temporal sampling for multi-point evaluation.

Sample markets at multiple time points before close to evaluate
how prediction quality degrades with uncertainty.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from data_loader.loader import Market, PriceData


@dataclass
class TemporalSample:
    """A single temporal sample from a market."""

    # Market identification
    market_ticker: str
    market: Market

    # Temporal information
    days_before_close: int
    sample_date: datetime
    close_date: datetime

    # Price data at sample time
    price_data: PriceData
    market_price: float  # The market's prediction (implied probability)

    # Ground truth
    outcome: float  # 1.0 for YES, 0.0 for NO

    # Context for prompting
    question: str = ""
    full_context: str = ""
    yes_means: str = ""
    no_means: str = ""

    @property
    def sample_id(self) -> str:
        """Unique identifier for this sample."""
        return f"{self.market_ticker}_T-{self.days_before_close}"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "market_ticker": self.market_ticker,
            "days_before_close": self.days_before_close,
            "sample_date": self.sample_date.isoformat(),
            "close_date": self.close_date.isoformat(),
            "market_price": self.market_price,
            "outcome": self.outcome,
            "question": self.question,
            "full_context": self.full_context,
            "yes_means": self.yes_means,
            "no_means": self.no_means,
        }


@dataclass
class TemporalSamplerConfig:
    """Configuration for temporal sampling."""

    # Days before close to sample
    temporal_days: list[int] = field(default_factory=lambda: [7, 3, 1])

    # Tolerance for finding price data (days)
    date_tolerance: int = 2

    # Whether to require exact date match
    require_exact_date: bool = False


class TemporalSampler:
    """Generate temporal samples from markets."""

    def __init__(self, config: Optional[TemporalSamplerConfig] = None):
        self.config = config or TemporalSamplerConfig()

    def sample_market(self, market: Market) -> list[TemporalSample]:
        """
        Generate temporal samples for a single market.

        Args:
            market: The market to sample from

        Returns:
            List of TemporalSample objects, one per temporal point where data exists
        """
        samples = []

        if not market.is_settled or market.outcome is None:
            return samples

        if not market.close_date:
            return samples

        for n_days in self.config.temporal_days:
            sample = self._create_sample(market, n_days)
            if sample is not None:
                samples.append(sample)

        return samples

    def sample_markets(self, markets: list[Market]) -> list[TemporalSample]:
        """
        Generate temporal samples for multiple markets.

        Args:
            markets: List of markets to sample from

        Returns:
            List of all TemporalSample objects
        """
        all_samples = []
        for market in markets:
            samples = self.sample_market(market)
            all_samples.extend(samples)
        return all_samples

    def _create_sample(self, market: Market, n_days: int) -> Optional[TemporalSample]:
        """Create a single temporal sample."""
        if not market.close_date or not market.daily_timeseries:
            return None

        target_date = market.close_date - timedelta(days=n_days)

        # Find price data for target date (with tolerance)
        price_data = self._find_price_data(market, target_date)

        if price_data is None:
            return None

        market_price = price_data.market_price
        if market_price is None:
            return None

        return TemporalSample(
            market_ticker=market.ticker,
            market=market,
            days_before_close=n_days,
            sample_date=datetime.strptime(price_data.date, "%Y-%m-%d"),
            close_date=market.close_date,
            price_data=price_data,
            market_price=market_price,
            outcome=market.outcome,  # type: ignore (we checked is_settled)
            question=market.question,
            full_context=market.full_context,
            yes_means=market.yes_means,
            no_means=market.no_means,
        )

    def _find_price_data(self, market: Market, target_date: datetime) -> Optional[PriceData]:
        """
        Find price data for a target date, with tolerance.

        Prefers exact match, then closest earlier date within tolerance.
        """
        target_str = target_date.strftime("%Y-%m-%d")

        # Try exact match first
        for pd in market.daily_timeseries:
            if pd.date == target_str:
                return pd

        if self.config.require_exact_date:
            return None

        # Find closest earlier date within tolerance
        best_match = None
        best_diff = None

        for pd in market.daily_timeseries:
            pd_date = datetime.strptime(pd.date, "%Y-%m-%d")

            # Only consider dates before or on target
            if pd_date > target_date:
                continue

            diff = (target_date - pd_date).days

            if diff <= self.config.date_tolerance:
                if best_diff is None or diff < best_diff:
                    best_match = pd
                    best_diff = diff

        return best_match

    def get_sampling_stats(self, samples: list[TemporalSample]) -> dict:
        """Get statistics about generated samples."""
        if not samples:
            return {"total_samples": 0}

        by_temporal = {}
        for s in samples:
            key = f"T-{s.days_before_close}"
            if key not in by_temporal:
                by_temporal[key] = []
            by_temporal[key].append(s)

        return {
            "total_samples": len(samples),
            "unique_markets": len(set(s.market_ticker for s in samples)),
            "by_temporal_point": {
                k: {
                    "count": len(v),
                    "mean_market_price": sum(s.market_price for s in v) / len(v),
                    "outcome_rate": sum(s.outcome for s in v) / len(v),
                }
                for k, v in by_temporal.items()
            },
        }
