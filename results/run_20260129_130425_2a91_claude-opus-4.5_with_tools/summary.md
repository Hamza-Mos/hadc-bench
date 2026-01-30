# Benchmark Run: run_20260129_130425_2a91_claude-opus-4.5_after_fix

Started: 2026-01-29T13:04:25.482405
Completed: 2026-01-29T13:30:19.557911
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
- tools_enabled: True
- n_samples: 750

## Results

Total predictions: 750
Beats market: 320/750 (42.7%)
Ties: 30/750 (4.0%)

### claude-opus-4.5 (agentic)

- Samples: 750
- Brier Score: 0.1652
- Brier Skill Score: -0.0685 (beats market: NO)
- ECE: 0.1340
- Accuracy: 0.776
- F1: 0.632

### Checkpoint Breakdown

- open_plus_1: Brier=0.1995, BSS=+0.1666, n=150
- pct_25: Brier=0.1735, BSS=+0.0592, n=150
- pct_50: Brier=0.1708, BSS=-0.1365, n=150
- pct_75: Brier=0.1434, BSS=-0.1648, n=150
- close_minus_1: Brier=0.1389, BSS=-0.8295, n=150
