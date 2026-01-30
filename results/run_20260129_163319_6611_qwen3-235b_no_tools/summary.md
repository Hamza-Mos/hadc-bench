# Benchmark Run: run_20260129_163319_6611_qwen3-235b_no_tools_after_fix

Started: 2026-01-29T16:33:19.872004
Completed: 2026-01-29T18:09:03.652351
Status: completed

## Configuration

- model_ids: ['qwen3-235b']
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
Beats market: 261/750 (34.8%)
Ties: 22/750 (2.9%)

### qwen3-235b (agentic)

- Samples: 750
- Brier Score: 0.2878
- Brier Skill Score: -0.8608 (beats market: NO)
- ECE: 0.2329
- Accuracy: 0.577
- F1: 0.258

### Checkpoint Breakdown

- open_plus_1: Brier=0.2798, BSS=-0.1684, n=150
- pct_25: Brier=0.3101, BSS=-0.6812, n=150
- pct_50: Brier=0.3034, BSS=-1.0184, n=150
- pct_75: Brier=0.2609, BSS=-1.1193, n=150
- close_minus_1: Brier=0.2847, BSS=-2.7500, n=150
