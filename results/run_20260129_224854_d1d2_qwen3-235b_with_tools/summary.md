# Benchmark Run: run_20260129_224854_d1d2_qwen3-235b_after_fix

Started: 2026-01-29T22:48:54.568067
Completed: 2026-01-30T01:44:42.375874
Status: completed

## Configuration

- model_ids: ['qwen3-235b']
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
Beats market: 267/750 (35.6%)
Ties: 13/750 (1.7%)

### qwen3-235b (agentic)

- Samples: 750
- Brier Score: 0.2498
- Brier Skill Score: -0.6154 (beats market: NO)
- ECE: 0.1951
- Accuracy: 0.633
- F1: 0.478

### Checkpoint Breakdown

- open_plus_1: Brier=0.2710, BSS=-0.1317, n=150
- pct_25: Brier=0.2585, BSS=-0.4014, n=150
- pct_50: Brier=0.2939, BSS=-0.9551, n=150
- pct_75: Brier=0.2485, BSS=-1.0185, n=150
- close_minus_1: Brier=0.1772, BSS=-1.3344, n=150
