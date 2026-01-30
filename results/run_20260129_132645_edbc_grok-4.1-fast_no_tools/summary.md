# Benchmark Run: run_20260129_132645_edbc_grok-4.1-fast_no_tools_after_fix

Started: 2026-01-29T13:26:45.680789
Completed: 2026-01-29T13:36:20.828439
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
- tools_enabled: False
- n_samples: 750

## Results

Total predictions: 750
Beats market: 307/750 (40.9%)
Ties: 16/750 (2.1%)

### grok-4.1-fast (agentic)

- Samples: 750
- Brier Score: 0.2874
- Brier Skill Score: -0.8587 (beats market: NO)
- ECE: 0.2483
- Accuracy: 0.589
- F1: 0.226

### Checkpoint Breakdown

- open_plus_1: Brier=0.2915, BSS=-0.2174, n=150
- pct_25: Brier=0.2819, BSS=-0.5282, n=150
- pct_50: Brier=0.3060, BSS=-1.0358, n=150
- pct_75: Brier=0.2831, BSS=-1.2995, n=150
- close_minus_1: Brier=0.2747, BSS=-2.6192, n=150
