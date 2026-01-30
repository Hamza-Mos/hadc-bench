# Benchmark Run: run_20260129_133019_f8c0_gemini-3-pro_after_fix

Started: 2026-01-29T13:30:19.787018
Completed: 2026-01-29T14:05:29.889623
Status: completed

## Configuration

- model_ids: ['gemini-3-pro']
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
Beats market: 372/750 (49.6%)
Ties: 19/750 (2.5%)

### gemini-3-pro (agentic)

- Samples: 750
- Brier Score: 0.2028
- Brier Skill Score: -0.3115 (beats market: NO)
- ECE: 0.1467
- Accuracy: 0.728
- F1: 0.547

### Checkpoint Breakdown

- open_plus_1: Brier=0.2597, BSS=-0.0845, n=150
- pct_25: Brier=0.2307, BSS=-0.2504, n=150
- pct_50: Brier=0.2063, BSS=-0.3726, n=150
- pct_75: Brier=0.1859, BSS=-0.5097, n=150
- close_minus_1: Brier=0.1316, BSS=-0.7338, n=150
