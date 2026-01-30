# Benchmark Run: run_20260129_133620_270c_intellect-3_no_tools_after_fix

Started: 2026-01-29T13:36:20.886309
Completed: 2026-01-29T14:06:29.414821
Status: completed

## Configuration

- model_ids: ['intellect-3']
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
Beats market: 207/750 (27.6%)
Ties: 11/750 (1.5%)

### intellect-3 (agentic)

- Samples: 750
- Brier Score: 0.2715
- Brier Skill Score: -0.7554 (beats market: NO)
- ECE: 0.2194
- Accuracy: 0.561
- F1: 0.338

### Checkpoint Breakdown

- open_plus_1: Brier=0.2855, BSS=-0.1924, n=150
- pct_25: Brier=0.2554, BSS=-0.3846, n=150
- pct_50: Brier=0.2628, BSS=-0.7483, n=150
- pct_75: Brier=0.2907, BSS=-1.3608, n=150
- close_minus_1: Brier=0.2630, BSS=-2.4648, n=150
