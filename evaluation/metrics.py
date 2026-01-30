"""
Evaluation metrics for forecasting performance.

Implements metrics from KalshiBench paper:
- Brier Score (prediction accuracy)
- Brier Skill Score (vs market baseline)
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Classification metrics (Accuracy, Precision, Recall, F1)
"""

from dataclasses import dataclass, field
from typing import Optional
import math


def brier_score(predictions: list[float], outcomes: list[float]) -> float:
    """
    Calculate Brier Score - Mean Squared Error between predictions and outcomes.

    BS = (1/N) * sum((p_i - o_i)^2)

    Lower is better (0 = perfect, 0.25 = random binary prediction).

    Args:
        predictions: List of predicted probabilities [0, 1]
        outcomes: List of actual outcomes (0 or 1)

    Returns:
        Brier Score
    """
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have same length")

    if not predictions:
        return 0.0

    squared_errors = [(p - o) ** 2 for p, o in zip(predictions, outcomes)]
    return sum(squared_errors) / len(squared_errors)


def brier_skill_score(
    model_predictions: list[float],
    baseline_predictions: list[float],
    outcomes: list[float],
) -> float:
    """
    Calculate Brier Skill Score - improvement over baseline.

    BSS = 1 - (BS_model / BS_baseline)

    BSS > 0: Model beats baseline
    BSS = 0: Model equals baseline
    BSS < 0: Model worse than baseline

    Args:
        model_predictions: Model's predicted probabilities
        baseline_predictions: Baseline (market) probabilities
        outcomes: Actual outcomes

    Returns:
        Brier Skill Score
    """
    bs_model = brier_score(model_predictions, outcomes)
    bs_baseline = brier_score(baseline_predictions, outcomes)

    if bs_baseline == 0:
        # Perfect baseline - can't do better
        return -float("inf") if bs_model > 0 else 0.0

    return 1 - (bs_model / bs_baseline)


def expected_calibration_error(
    predictions: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE = sum(|bin_i| / N * |accuracy_i - confidence_i|)

    Lower is better (0 = perfectly calibrated).

    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes
        n_bins: Number of calibration bins

    Returns:
        ECE value
    """
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have same length")

    if not predictions:
        return 0.0

    n = len(predictions)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]

        # Get samples in this bin
        in_bin = [
            (p, o) for p, o in zip(predictions, outcomes)
            if lower <= p < upper or (i == n_bins - 1 and p == upper)
        ]

        if not in_bin:
            continue

        bin_size = len(in_bin)
        bin_accuracy = sum(o for _, o in in_bin) / bin_size
        bin_confidence = sum(p for p, _ in in_bin) / bin_size

        ece += (bin_size / n) * abs(bin_accuracy - bin_confidence)

    return ece


def maximum_calibration_error(
    predictions: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    MCE = max(|accuracy_i - confidence_i|)

    Lower is better.

    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes
        n_bins: Number of calibration bins

    Returns:
        MCE value
    """
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have same length")

    if not predictions:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    max_error = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]

        # Get samples in this bin
        in_bin = [
            (p, o) for p, o in zip(predictions, outcomes)
            if lower <= p < upper or (i == n_bins - 1 and p == upper)
        ]

        if not in_bin:
            continue

        bin_accuracy = sum(o for _, o in in_bin) / len(in_bin)
        bin_confidence = sum(p for p, _ in in_bin) / len(in_bin)

        error = abs(bin_accuracy - bin_confidence)
        max_error = max(max_error, error)

    return max_error


def accuracy(predictions: list[float], outcomes: list[float], threshold: float = 0.5) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        threshold: Classification threshold

    Returns:
        Accuracy (0 to 1)
    """
    if not predictions:
        return 0.0

    correct = sum(
        1 for p, o in zip(predictions, outcomes)
        if (p >= threshold and o == 1) or (p < threshold and o == 0)
    )
    return correct / len(predictions)


def precision(predictions: list[float], outcomes: list[float], threshold: float = 0.5) -> float:
    """
    Calculate precision for YES predictions.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        threshold: Classification threshold

    Returns:
        Precision (0 to 1)
    """
    predicted_positive = [(p, o) for p, o in zip(predictions, outcomes) if p >= threshold]

    if not predicted_positive:
        return 0.0

    true_positive = sum(1 for _, o in predicted_positive if o == 1)
    return true_positive / len(predicted_positive)


def recall(predictions: list[float], outcomes: list[float], threshold: float = 0.5) -> float:
    """
    Calculate recall for YES outcomes.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        threshold: Classification threshold

    Returns:
        Recall (0 to 1)
    """
    actual_positive = [(p, o) for p, o in zip(predictions, outcomes) if o == 1]

    if not actual_positive:
        return 0.0

    true_positive = sum(1 for p, _ in actual_positive if p >= threshold)
    return true_positive / len(actual_positive)


def f1_score(predictions: list[float], outcomes: list[float], threshold: float = 0.5) -> float:
    """
    Calculate F1 score.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        threshold: Classification threshold

    Returns:
        F1 score (0 to 1)
    """
    p = precision(predictions, outcomes, threshold)
    r = recall(predictions, outcomes, threshold)

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)


@dataclass
class BenchmarkMetrics:
    """Container for all benchmark metrics."""

    # Sample info
    n_samples: int = 0

    # Probability metrics
    brier_score: float = 0.0
    brier_skill_score: Optional[float] = None  # Only if baseline available

    # Calibration metrics
    ece: float = 0.0
    mce: float = 0.0

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Parse success rate
    parse_success_rate: float = 0.0

    # Breakdown by temporal point
    by_temporal: dict = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        model_predictions: list[float],
        market_predictions: list[float],
        outcomes: list[float],
        parse_successes: Optional[list[bool]] = None,
        n_bins: int = 10,
    ) -> "BenchmarkMetrics":
        """
        Compute all metrics from predictions.

        Args:
            model_predictions: Model's predicted probabilities (confidence in YES)
            market_predictions: Market's implied probabilities
            outcomes: Actual outcomes (1 for YES, 0 for NO)
            parse_successes: Whether each prediction was successfully parsed
            n_bins: Number of bins for calibration metrics

        Returns:
            BenchmarkMetrics with all computed values
        """
        n = len(model_predictions)

        if n == 0:
            return cls()

        # Parse success rate
        if parse_successes:
            parse_rate = sum(parse_successes) / len(parse_successes)
        else:
            parse_rate = 1.0

        return cls(
            n_samples=n,
            brier_score=brier_score(model_predictions, outcomes),
            brier_skill_score=brier_skill_score(model_predictions, market_predictions, outcomes),
            ece=expected_calibration_error(model_predictions, outcomes, n_bins),
            mce=maximum_calibration_error(model_predictions, outcomes, n_bins),
            accuracy=accuracy(model_predictions, outcomes),
            precision=precision(model_predictions, outcomes),
            recall=recall(model_predictions, outcomes),
            f1=f1_score(model_predictions, outcomes),
            parse_success_rate=parse_rate,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "brier_score": self.brier_score,
            "brier_skill_score": self.brier_skill_score,
            "ece": self.ece,
            "mce": self.mce,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "parse_success_rate": self.parse_success_rate,
            "by_temporal": self.by_temporal,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        bss_str = f"{self.brier_skill_score:+.3f}" if self.brier_skill_score else "N/A"
        beats_market = "YES" if self.brier_skill_score and self.brier_skill_score > 0 else "NO"

        return f"""Benchmark Results (n={self.n_samples})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Probability Metrics:
  Brier Score:       {self.brier_score:.4f} (lower is better)
  Brier Skill Score: {bss_str} (>0 beats market)
  Beats Market:      {beats_market}

Calibration Metrics:
  ECE: {self.ece:.4f} (lower is better)
  MCE: {self.mce:.4f}

Classification (threshold=0.5):
  Accuracy:  {self.accuracy:.3f}
  Precision: {self.precision:.3f}
  Recall:    {self.recall:.3f}
  F1:        {self.f1:.3f}

Parse Success Rate: {self.parse_success_rate:.1%}
"""
