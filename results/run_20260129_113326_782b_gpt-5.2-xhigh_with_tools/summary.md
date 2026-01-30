# Benchmark Run: run_20260129_113326_782b_gpt-5.2-xhigh_after_fix

Started: 2026-01-29T11:33:26.127520
Completed: 2026-01-29T11:51:44.217764
Status: completed

## Configuration

- model_ids: ['gpt-5.2-xhigh']
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
Beats market: 300/750 (40.0%)
Ties: 27/750 (3.6%)

### gpt-5.2-xhigh (agentic)

- Samples: 750
- Brier Score: 0.1936
- Brier Skill Score: -0.2521 (beats market: NO)
- ECE: 0.1110
- Accuracy: 0.709
- F1: 0.476

### Checkpoint Breakdown

- open_plus_1: Brier=0.2191, BSS=+0.0849, n=150
- pct_25: Brier=0.2020, BSS=-0.0949, n=150
- pct_50: Brier=0.2020, BSS=-0.3435, n=150
- pct_75: Brier=0.1964, BSS=-0.5951, n=150
- close_minus_1: Brier=0.1488, BSS=-0.9601, n=150
