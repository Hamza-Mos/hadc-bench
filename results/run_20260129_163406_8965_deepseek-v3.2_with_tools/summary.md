# Benchmark Run: run_20260129_163406_8965_deepseek-v3.2_after_fix

Started: 2026-01-29T16:34:06.756210
Completed: 2026-01-29T17:25:05.645230
Status: completed

## Configuration

- model_ids: ['deepseek-v3.2']
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
Beats market: 249/750 (33.2%)
Ties: 14/750 (1.9%)

### deepseek-v3.2 (agentic)

- Samples: 750
- Brier Score: 0.2331
- Brier Skill Score: -0.5074 (beats market: NO)
- ECE: 0.1962
- Accuracy: 0.625
- F1: 0.484

### Checkpoint Breakdown

- open_plus_1: Brier=0.2547, BSS=-0.0639, n=150
- pct_25: Brier=0.2627, BSS=-0.4242, n=150
- pct_50: Brier=0.2541, BSS=-0.6904, n=150
- pct_75: Brier=0.2303, BSS=-0.8702, n=150
- close_minus_1: Brier=0.1638, BSS=-1.1576, n=150
