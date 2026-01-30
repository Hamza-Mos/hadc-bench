# Benchmark Run: run_20260129_140629_4f2b_deepseek-v3.2_no_tools_after_fix

Started: 2026-01-29T14:06:29.462225
Completed: 2026-01-29T14:13:59.906731
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
- tools_enabled: False
- n_samples: 750

## Results

Total predictions: 750
Beats market: 214/750 (28.5%)
Ties: 14/750 (1.9%)

### deepseek-v3.2 (agentic)

- Samples: 750
- Brier Score: 0.2625
- Brier Skill Score: -0.6972 (beats market: NO)
- ECE: 0.1920
- Accuracy: 0.565
- F1: 0.279

### Checkpoint Breakdown

- open_plus_1: Brier=0.2673, BSS=-0.1165, n=150
- pct_25: Brier=0.2792, BSS=-0.5137, n=150
- pct_50: Brier=0.2687, BSS=-0.7874, n=150
- pct_75: Brier=0.2541, BSS=-1.0636, n=150
- close_minus_1: Brier=0.2431, BSS=-2.2020, n=150
