#!/usr/bin/env python3
"""
Fetch diverse benchmark dataset using Kalshi native categories (v2)
- Fetches top N markets PER SERIES to guarantee diversity
- Proper logging to track where markets are lost
- Fallback logic when candlesticks unavailable
"""

import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PARALLEL_CATEGORIES = 2  # Process N categories at once

TARGET_PER_CATEGORY = 30
MARKETS_PER_SERIES = 5  # Fetch top 5 markets from each series
MIN_CLOSE_DATE = "2025-10-01"
MAX_CLOSE_DATE = "2026-01-28"
MIN_MARKET_VOLUME = 5000  # Minimum total volume for market to be included
MIN_CHECKPOINTS_WITH_VOLUME = 2  # At least N checkpoints must have daily volume > 0

CATEGORY_MAPPING = {
    "Politics": "Politics/Elections",
    "Elections": "Politics/Elections",
    "Economics": "MacroEconomics",
    "Financials": "Financial",
    "Crypto": "Financial",
    "Science and Technology": "Science/Health/Tech",
    "Health": "Science/Health/Tech",
    "Sports": "Sports/Entertainment",
    "Entertainment": "Sports/Entertainment",
}

CHECKPOINTS = [
    ("open_plus_1", None),
    ("pct_25", 0.25),
    ("pct_50", 0.50),
    ("pct_75", 0.75),
    ("close_minus_1", None),
]

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def log(msg, indent=0):
    """Simple logging with indentation"""
    prefix = "  " * indent
    print(f"{prefix}{msg}")


def fetch_series_by_category() -> dict:
    """Fetch all series, group by our categories"""
    log("📦 Fetching all series...")

    session = requests.Session()
    series_by_category = defaultdict(list)
    cursor = None
    total = 0

    while True:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor

        response = session.get(f"{BASE_URL}/series", params=params)
        response.raise_for_status()
        data = response.json()

        batch = data.get("series", [])
        if not batch:
            break

        total += len(batch)

        for s in batch:
            kalshi_cat = s.get("category", "")
            our_cat = CATEGORY_MAPPING.get(kalshi_cat)
            if our_cat:
                series_by_category[our_cat].append({
                    "ticker": s["ticker"],
                    "title": s.get("title", ""),
                    "kalshi_category": kalshi_cat,
                })

        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.02)

    log(f"   Total series from API: {total}")
    for cat, series_list in series_by_category.items():
        log(f"   {cat}: {len(series_list)} series", 1)

    return dict(series_by_category)


def fetch_markets_for_series(session, series_ticker: str) -> list:
    """Fetch resolved markets for a single series within date range"""
    min_ts = int(datetime.strptime(MIN_CLOSE_DATE, "%Y-%m-%d").timestamp())
    max_ts = int(datetime.strptime(MAX_CLOSE_DATE, "%Y-%m-%d").timestamp())

    params = {
        "limit": 100,
        "series_ticker": series_ticker,
        "min_close_ts": min_ts,
        "max_close_ts": max_ts,
    }

    try:
        response = session.get(f"{BASE_URL}/markets", params=params)
        response.raise_for_status()
        data = response.json()

        markets = []
        for m in data.get("markets", []):
            if m.get("result") and m.get("volume", 0) >= MIN_MARKET_VOLUME:
                markets.append(m)

        # Sort by volume, return top N
        markets.sort(key=lambda x: x.get("volume", 0), reverse=True)
        return markets[:MARKETS_PER_SERIES]

    except Exception as e:
        return []


def fetch_candlesticks(session, series_ticker: str, market_ticker: str) -> list:
    """Fetch daily candlesticks for a market"""
    url = f"{BASE_URL}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    params = {
        "start_ts": int((datetime.now() - timedelta(days=365)).timestamp()),
        "end_ts": int(datetime.now().timestamp()),
        "period_interval": 1440,
    }

    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        return response.json().get("candlesticks", [])
    except:
        return []


def get_checkpoint_date(open_dt: datetime, close_dt: datetime, checkpoint: tuple) -> str:
    name, pct = checkpoint
    duration = close_dt - open_dt

    if name == "open_plus_1":
        cp_dt = open_dt + timedelta(days=1)
    elif name == "close_minus_1":
        cp_dt = close_dt - timedelta(days=1)
    else:
        cp_dt = open_dt + (duration * pct)

    cp_dt = max(cp_dt, open_dt + timedelta(days=1))
    cp_dt = min(cp_dt, close_dt - timedelta(days=1))

    return cp_dt.strftime("%Y-%m-%d")


def get_price_at_date(candlesticks: list, target_date: str) -> dict | None:
    target = datetime.strptime(target_date, "%Y-%m-%d")

    prices_by_date = {}
    for candle in candlesticks:
        ts = candle.get("end_period_ts")
        if ts:
            date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            prices_by_date[date] = candle

    for days_back in range(7):
        check_date = (target - timedelta(days=days_back)).strftime("%Y-%m-%d")
        if check_date in prices_by_date:
            return prices_by_date[check_date]

    return None


print_lock = threading.Lock()


def process_category(category: str, series_list: list) -> tuple:
    """Process a single category - can be run in parallel"""
    session = requests.Session()

    stats = {
        "series_checked": 0,
        "series_with_markets": 0,
        "markets_found": 0,
        "markets_no_candlesticks": 0,
        "markets_no_checkpoints": 0,
        "markets_low_checkpoint_volume": 0,
        "markets_success": 0,
        "samples_created": 0,
    }

    with print_lock:
        log(f"\n📊 {category} ({len(series_list)} series available)")

    cat_samples = []
    markets_collected = 0
    series_used = set()

    for series_info in series_list:
        if markets_collected >= TARGET_PER_CATEGORY:
            break

        series_ticker = series_info["ticker"]
        stats["series_checked"] += 1

        # Fetch markets for this series
        markets = fetch_markets_for_series(session, series_ticker)
        time.sleep(0.04)  # Slightly longer sleep for parallel requests

        if not markets:
            continue

        stats["series_with_markets"] += 1
        stats["markets_found"] += len(markets)

        # Try each market in this series
        for market in markets:
            if markets_collected >= TARGET_PER_CATEGORY:
                break

            ticker = market.get("ticker")
            open_time = market.get("open_time", "")
            close_time = market.get("close_time", "")

            if not (open_time and close_time):
                continue

            try:
                open_dt = datetime.fromisoformat(open_time.replace("Z", "+00:00")).replace(tzinfo=None)
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00")).replace(tzinfo=None)
            except:
                continue

            # Fetch candlesticks
            candlesticks = fetch_candlesticks(session, series_ticker, ticker)
            time.sleep(0.05)  # Slightly longer sleep for parallel requests

            if not candlesticks:
                stats["markets_no_candlesticks"] += 1
                with print_lock:
                    log(f"   [{category[:8]}] ⚠️  No candlesticks: {ticker}", 1)
                continue

            # Build samples for each checkpoint - MUST have all 5
            market_samples = []
            all_checkpoints_valid = True

            for checkpoint in CHECKPOINTS:
                cp_name, cp_pct = checkpoint
                cp_date = get_checkpoint_date(open_dt, close_dt, checkpoint)
                price_data = get_price_at_date(candlesticks, cp_date)

                if not price_data:
                    all_checkpoints_valid = False
                    break

                sample = {
                    "checkpoint_name": cp_name,
                    "checkpoint_pct": cp_pct,
                    "checkpoint_date": cp_date,
                    "price_at_checkpoint": price_data,
                    "ticker": ticker,
                    "series_ticker": series_ticker,
                    "title": market.get("title"),
                    "category": category,
                    "subtitle": market.get("subtitle"),
                    "yes_sub_title": market.get("yes_sub_title"),
                    "no_sub_title": market.get("no_sub_title"),
                    "rules_primary": market.get("rules_primary"),
                    "result": market.get("result"),
                    "status": market.get("status"),
                    "open_time": open_time,
                    "close_time": close_time,
                    "volume": market.get("volume"),
                    "open_interest": market.get("open_interest"),
                    "duration_days": (close_dt - open_dt).days,
                }
                market_samples.append(sample)

            if all_checkpoints_valid and len(market_samples) == 5:
                # Check how many checkpoints have daily volume > 0
                checkpoints_with_volume = sum(
                    1 for s in market_samples
                    if s["price_at_checkpoint"].get("volume", 0) > 0
                )

                if checkpoints_with_volume >= MIN_CHECKPOINTS_WITH_VOLUME:
                    cat_samples.extend(market_samples)
                    markets_collected += 1
                    series_used.add(series_ticker)
                    stats["markets_success"] += 1
                    stats["samples_created"] += len(market_samples)
                    with print_lock:
                        log(f"   [{category[:8]}] ✓ {ticker} (5/5 cp, {checkpoints_with_volume}/5 vol)", 1)
                else:
                    stats["markets_low_checkpoint_volume"] += 1
                    with print_lock:
                        log(f"   [{category[:8]}] ⚠️  Low checkpoint volume: {ticker}", 1)
            else:
                stats["markets_no_checkpoints"] += 1
                with print_lock:
                    log(f"   [{category[:8]}] ⚠️  Incomplete checkpoints: {ticker}", 1)

    with print_lock:
        log(f"   [{category[:8]}] → {markets_collected} markets from {len(series_used)} series")

    return category, cat_samples, stats


def build_diverse_dataset(series_by_category: dict) -> list:
    """Build dataset by fetching from each series to ensure diversity"""
    log(f"\n🔨 Building dataset (target: {TARGET_PER_CATEGORY} per category)")
    log(f"   Date range: {MIN_CLOSE_DATE} to {MAX_CLOSE_DATE}")
    log(f"   Parallel categories: {PARALLEL_CATEGORIES}")

    all_samples = []
    all_stats = {}

    # Process categories in parallel
    with ThreadPoolExecutor(max_workers=PARALLEL_CATEGORIES) as executor:
        futures = {
            executor.submit(process_category, cat, series_list): cat
            for cat, series_list in series_by_category.items()
        }

        for future in as_completed(futures):
            category, cat_samples, stats = future.result()
            all_samples.extend(cat_samples)
            all_stats[category] = stats

    # Print detailed stats
    log(f"\n📈 Detailed Statistics:")
    for category, s in all_stats.items():
        log(f"\n   {category}:")
        log(f"      Series checked: {s['series_checked']}", 1)
        log(f"      Series with markets: {s['series_with_markets']}", 1)
        log(f"      Markets found: {s['markets_found']}", 1)
        log(f"      Markets no candlesticks: {s['markets_no_candlesticks']}", 1)
        log(f"      Markets low checkpoint volume: {s['markets_low_checkpoint_volume']}", 1)
        log(f"      Markets no checkpoints: {s['markets_no_checkpoints']}", 1)
        log(f"      Markets success: {s['markets_success']}", 1)
        log(f"      Samples created: {s['samples_created']}", 1)

    return all_samples


def main():
    print("=" * 60)
    print("Fetch Diverse Benchmark Dataset v2")
    print("=" * 60)

    # Step 1: Get all series by category
    series_by_category = fetch_series_by_category()

    if not series_by_category:
        print("No series found!")
        return

    # Step 2: Build diverse dataset
    samples = build_diverse_dataset(series_by_category)

    # Summary
    by_cat = defaultdict(int)
    by_cp = defaultdict(int)
    unique_markets = set()
    unique_series = set()

    for s in samples:
        by_cat[s["category"]] += 1
        by_cp[s["checkpoint_name"]] += 1
        unique_markets.add(s["ticker"])
        unique_series.add(s["series_ticker"])

    print(f"\n{'=' * 60}")
    print(f"Total samples: {len(samples)}")
    print(f"Unique markets: {len(unique_markets)}")
    print(f"Unique series: {len(unique_series)}")

    print("\nBy category:")
    for cat, count in sorted(by_cat.items()):
        markets = count // 5 if count >= 5 else count
        print(f"  {cat}: {count} samples (~{markets} markets)")

    print("\nBy checkpoint:")
    for cp, count in sorted(by_cp.items()):
        print(f"  {cp}: {count}")

    # Sample titles
    print("\nSample titles by category:")
    shown = defaultdict(int)
    for s in samples:
        cat = s["category"]
        if shown[cat] < 3 and s["checkpoint_name"] == "pct_50":
            print(f"  [{cat}] {s['series_ticker']}: {s['title']}")
            shown[cat] += 1

    # Save with version number
    output_dir = Path("dataset")
    output_dir.mkdir(exist_ok=True)

    # Find next available version number
    version = 1
    while (output_dir / f"benchmark_dataset_v{version}.json").exists():
        version += 1

    output = {
        "created_at": datetime.now().isoformat(),
        "version": version,
        "min_close_date": MIN_CLOSE_DATE,
        "max_close_date": MAX_CLOSE_DATE,
        "target_per_category": TARGET_PER_CATEGORY,
        "markets_per_series": MARKETS_PER_SERIES,
        "min_market_volume": MIN_MARKET_VOLUME,
        "min_checkpoints_with_volume": MIN_CHECKPOINTS_WITH_VOLUME,
        "categories": list(set(CATEGORY_MAPPING.values())),
        "checkpoints": [c[0] for c in CHECKPOINTS],
        "total_samples": len(samples),
        "unique_markets": len(unique_markets),
        "unique_series": len(unique_series),
        "by_category": dict(by_cat),
        "by_checkpoint": dict(by_cp),
        "samples": samples,
    }

    output_file = output_dir / f"benchmark_dataset_v{version}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
