# Benchmark Run: run_20260129_140530_45fb_grok-4.1-fast_after_fix

Started: 2026-01-29T14:05:30.034389
Completed: 2026-01-29T14:21:11.490230
Status: completed

## Configuration

- model_ids: ['grok-4.1-fast']
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
Beats market: 366/750 (48.8%)
Ties: 31/750 (4.1%)

### grok-4.1-fast (agentic)

- Samples: 750
- Brier Score: 0.1960
- Brier Skill Score: -0.2671 (beats market: NO)
- ECE: 0.1271
- Accuracy: 0.736
- F1: 0.533

### Checkpoint Breakdown

- open_plus_1: Brier=0.2401, BSS=-0.0030, n=150
- pct_25: Brier=0.1976, BSS=-0.0711, n=150
- pct_50: Brier=0.1976, BSS=-0.3143, n=150
- pct_75: Brier=0.1995, BSS=-0.6208, n=150
- close_minus_1: Brier=0.1449, BSS=-0.9089, n=150
