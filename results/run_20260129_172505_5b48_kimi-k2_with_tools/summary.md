# Benchmark Run: run_20260129_172505_5b48_kimi-k2_after_fix

Started: 2026-01-29T17:25:05.772719
Completed: 2026-01-29T19:13:25.136315
Status: completed

## Configuration

- model_ids: ['kimi-k2']
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
Beats market: 306/750 (40.8%)
Ties: 25/750 (3.3%)

### kimi-k2 (agentic)

- Samples: 750
- Brier Score: 0.2113
- Brier Skill Score: -0.3664 (beats market: NO)
- ECE: 0.1577
- Accuracy: 0.695
- F1: 0.528

### Checkpoint Breakdown

- open_plus_1: Brier=0.2369, BSS=+0.0106, n=150
- pct_25: Brier=0.2052, BSS=-0.1126, n=150
- pct_50: Brier=0.2407, BSS=-0.6013, n=150
- pct_75: Brier=0.2354, BSS=-0.9117, n=150
- close_minus_1: Brier=0.1384, BSS=-0.8231, n=150
