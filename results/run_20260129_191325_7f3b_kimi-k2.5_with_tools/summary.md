# Benchmark Run: run_20260129_191325_7f3b_kimi-k2.5_after_fix

Started: 2026-01-29T19:13:25.259646
Completed: 2026-01-29T20:17:17.898721
Status: completed

## Configuration

- model_ids: ['kimi-k2.5']
- strategies: ['agentic']
- benchmark_dataset: categorized_data/benchmark_dataset_v2.json
- checkpoints: ['open_plus_1', 'pct_25', 'pct_50', 'pct_75', 'close_minus_1']
- categories: None
- max_iterations: 100
- max_search_results: 5
- parallelism: 50
- tools_enabled: True
- n_samples: 750

## Results

Total predictions: 750
Beats market: 297/750 (39.6%)
Ties: 18/750 (2.4%)

### kimi-k2.5 (agentic)

- Samples: 750
- Brier Score: 0.1877
- Brier Skill Score: -0.2138 (beats market: NO)
- ECE: 0.1246
- Accuracy: 0.720
- F1: 0.525

### Checkpoint Breakdown

- open_plus_1: Brier=0.2112, BSS=+0.1180, n=150
- pct_25: Brier=0.1884, BSS=-0.0215, n=150
- pct_50: Brier=0.1971, BSS=-0.3114, n=150
- pct_75: Brier=0.1944, BSS=-0.5791, n=150
- close_minus_1: Brier=0.1474, BSS=-0.9416, n=150
