"""
Market data loader for Kalshi prediction markets.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class PriceData:
    """OHLC price data for a single day."""

    date: str
    timestamp: int
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    mean: Optional[float] = None
    volume_contracts: int = 0
    open_interest_contracts: int = 0

    # Yes ask/bid data (more complete than price)
    yes_ask_close: Optional[float] = None
    yes_bid_close: Optional[float] = None

    @property
    def market_price(self) -> Optional[float]:
        """
        Get the best available market price.
        Prefer yes_ask_close as it represents the price to buy YES.
        Falls back to close price or midpoint.
        """
        if self.yes_ask_close is not None:
            return self.yes_ask_close
        if self.close is not None:
            return self.close
        if self.yes_ask_close is not None and self.yes_bid_close is not None:
            return (self.yes_ask_close + self.yes_bid_close) / 2
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriceData":
        """Create PriceData from raw timeseries entry."""
        price_data = data.get("price", {})
        yes_ask = data.get("yes_ask", {})
        yes_bid = data.get("yes_bid", {})
        volume = data.get("volume", {})
        open_interest = data.get("open_interest", {})

        return cls(
            date=data["date"],
            timestamp=data["timestamp"],
            open=price_data.get("open"),
            high=price_data.get("high"),
            low=price_data.get("low"),
            close=price_data.get("close"),
            mean=price_data.get("mean"),
            volume_contracts=volume.get("contracts", 0),
            open_interest_contracts=open_interest.get("contracts", 0),
            yes_ask_close=yes_ask.get("close"),
            yes_bid_close=yes_bid.get("close"),
        )


@dataclass
class Market:
    """Representation of a Kalshi prediction market."""

    # Core identification
    ticker: str
    event_ticker: Optional[str] = None

    # Question and context
    question: str = ""
    full_context: str = ""
    yes_means: str = ""
    no_means: str = ""
    resolution_rules: str = ""
    additional_rules: str = ""

    # Status and result
    status: str = ""
    result: Optional[str] = None  # "yes" or "no" for settled markets
    what_happened: Optional[str] = None  # Human-readable outcome

    # Categorization
    category: str = ""
    tags: list[str] = field(default_factory=list)

    # Timing
    open_date: Optional[datetime] = None
    close_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None

    # Trading stats
    total_volume_contracts: int = 0
    volume_24h_contracts: int = 0
    open_interest_contracts: int = 0

    # Price history
    daily_timeseries: list[PriceData] = field(default_factory=list)

    # Raw data reference
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_settled(self) -> bool:
        """Check if market has a final result."""
        return self.status == "finalized" and self.result in ("yes", "no")

    @property
    def outcome(self) -> Optional[float]:
        """Return outcome as float: 1.0 for yes, 0.0 for no, None if unsettled."""
        if not self.is_settled:
            return None
        return 1.0 if self.result == "yes" else 0.0

    def get_price_at_date(self, date_str: str) -> Optional[PriceData]:
        """Get price data for a specific date."""
        for pd in self.daily_timeseries:
            if pd.date == date_str:
                return pd
        return None

    def get_price_n_days_before_close(self, n_days: int) -> Optional[PriceData]:
        """
        Get price data N days before market close.

        Args:
            n_days: Number of days before close (e.g., 7 for T-7)

        Returns:
            PriceData for that date, or None if not available
        """
        if not self.close_date or not self.daily_timeseries:
            return None

        from datetime import timedelta

        target_date = self.close_date - timedelta(days=n_days)
        target_str = target_date.strftime("%Y-%m-%d")

        # Find exact match or closest earlier date
        best_match = None
        for pd in self.daily_timeseries:
            pd_date = datetime.strptime(pd.date, "%Y-%m-%d")
            if pd_date <= target_date:
                if best_match is None or pd_date > datetime.strptime(best_match.date, "%Y-%m-%d"):
                    best_match = pd

        return best_match

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Market":
        """Create Market from JSON data."""
        bet_exp = data.get("bet_explanation", {})
        metadata = data.get("market_metadata", {})
        stats = data.get("trading_stats", {})
        timeseries_raw = data.get("daily_timeseries", [])

        # Parse dates
        def parse_date(date_str: Optional[str]) -> Optional[datetime]:
            if not date_str:
                return None
            try:
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S UTC")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    return None

        # Parse timeseries
        timeseries = [PriceData.from_dict(entry) for entry in timeseries_raw]

        return cls(
            ticker=metadata.get("ticker", ""),
            event_ticker=metadata.get("event_ticker"),
            question=bet_exp.get("question", ""),
            full_context=bet_exp.get("full_context", ""),
            yes_means=bet_exp.get("yes_means", ""),
            no_means=bet_exp.get("no_means", ""),
            resolution_rules=bet_exp.get("resolution_rules", ""),
            additional_rules=bet_exp.get("additional_rules", ""),
            status=metadata.get("status", ""),
            result=metadata.get("result"),
            what_happened=bet_exp.get("what_happened"),
            category=metadata.get("category", ""),
            tags=metadata.get("tags", []),
            open_date=parse_date(metadata.get("open_date")),
            close_date=parse_date(metadata.get("close_date")),
            expiration_date=parse_date(metadata.get("expiration_date")),
            total_volume_contracts=stats.get("total_volume_contracts", 0),
            volume_24h_contracts=stats.get("volume_24h_contracts", 0),
            open_interest_contracts=stats.get("open_interest_contracts", 0),
            daily_timeseries=timeseries,
            raw_data=data,
        )


class MarketLoader:
    """Load and manage Kalshi market data."""

    def __init__(self, data_dir: str | Path = "kalshi_data"):
        self.data_dir = Path(data_dir)
        self.markets_dir = self.data_dir / "markets"
        self._markets_cache: Optional[list[Market]] = None

    def load_market(self, ticker: str) -> Optional[Market]:
        """Load a single market by ticker."""
        market_file = self.markets_dir / f"{ticker}.json"
        if not market_file.exists():
            return None

        with open(market_file) as f:
            data = json.load(f)

        return Market.from_json(data)

    def load_all_markets(self, use_cache: bool = True) -> list[Market]:
        """Load all markets from individual JSON files."""
        if use_cache and self._markets_cache is not None:
            return self._markets_cache

        markets = []
        if not self.markets_dir.exists():
            return markets

        for market_file in sorted(self.markets_dir.glob("*.json")):
            with open(market_file) as f:
                data = json.load(f)
            markets.append(Market.from_json(data))

        self._markets_cache = markets
        return markets

    def load_summary(self) -> dict[str, Any]:
        """Load the summary.json file."""
        summary_file = self.data_dir / "summary.json"
        if not summary_file.exists():
            return {}

        with open(summary_file) as f:
            return json.load(f)

    def get_market_count(self) -> int:
        """Get total number of markets without loading all data."""
        summary = self.load_summary()
        return summary.get("total_markets", 0)

    def clear_cache(self):
        """Clear the markets cache."""
        self._markets_cache = None
