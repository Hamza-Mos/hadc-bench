# Benchmark Run: run_20260129_130428_55de_claude-opus-4.5_no_tools_after_fix

Started: 2026-01-29T13:04:28.883243
Completed: 2026-01-29T13:08:21.480539
Status: completed

## Configuration

- model_ids: ['claude-opus-4.5']
- strategies: ['agentic']
- benchmark_dataset: categorized_data/benchmark_dataset_v2.json
- checkpoints: ['open_plus_1', 'pct_25', 'pct_50', 'pct_75', 'close_minus_1']
- categories: None
- max_iterations: 100
- max_search_results: 5
- parallelism: 50
- tools_enabled: False
- n_samples: 750

## Results

Total predictions: 750
Beats market: 221/750 (29.5%)
Ties: 11/750 (1.5%)

### claude-opus-4.5 (agentic)

- Samples: 750
- Brier Score: 0.2456
- Brier Skill Score: -0.5880 (beats market: NO)
- ECE: 0.1638
- Accuracy: 0.576
- F1: 0.321

### Checkpoint Breakdown

- open_plus_1: Brier=0.2449, BSS=-0.0227, n=150
- pct_25: Brier=0.2433, BSS=-0.3191, n=150
- pct_50: Brier=0.2520, BSS=-0.6764, n=150
- pct_75: Brier=0.2439, BSS=-0.9811, n=150
- close_minus_1: Brier=0.2438, BSS=-2.2122, n=150
