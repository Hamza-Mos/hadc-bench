# Benchmark Run: run_20260129_224908_bce7_trinity-large-preview_after_fix

Started: 2026-01-29T22:49:08.457198
Completed: 2026-01-30T05:30:57.959732
Status: completed

## Configuration

- model_ids: ['trinity-large-preview']
- strategies: ['agentic']
- benchmark_dataset: categorized_data/benchmark_dataset_v2.json
- checkpoints: ['open_plus_1', 'pct_25', 'pct_50', 'pct_75', 'close_minus_1']
- categories: None
- max_iterations: 100
- max_search_results: 5
- parallelism: 20
- tools_enabled: True
- n_samples: 750

## Results

Total predictions: 750
Beats market: 290/750 (38.7%)
Ties: 12/750 (1.6%)

### trinity-large-preview (agentic)

- Samples: 750
- Brier Score: 0.2313
- Brier Skill Score: -0.4957 (beats market: NO)
- ECE: 0.1835
- Accuracy: 0.661
- F1: 0.512

### Checkpoint Breakdown

- open_plus_1: Brier=0.2702, BSS=-0.1284, n=150
- pct_25: Brier=0.2352, BSS=-0.2749, n=150
- pct_50: Brier=0.2377, BSS=-0.5812, n=150
- pct_75: Brier=0.2074, BSS=-0.6844, n=150
- close_minus_1: Brier=0.2062, BSS=-1.7160, n=150
