# Benchmark Run: run_20260129_142111_5503_intellect-3_after_fix

Started: 2026-01-29T14:21:11.664678
Completed: 2026-01-29T15:40:17.437804
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
- tools_enabled: True
- n_samples: 750

## Results

Total predictions: 750
Beats market: 250/750 (33.3%)
Ties: 13/750 (1.7%)

### intellect-3 (agentic)

- Samples: 750
- Brier Score: 0.2362
- Brier Skill Score: -0.5275 (beats market: NO)
- ECE: 0.1805
- Accuracy: 0.656
- F1: 0.486

### Checkpoint Breakdown

- open_plus_1: Brier=0.2532, BSS=-0.0574, n=150
- pct_25: Brier=0.2644, BSS=-0.4334, n=150
- pct_50: Brier=0.2467, BSS=-0.6413, n=150
- pct_75: Brier=0.2091, BSS=-0.6984, n=150
- close_minus_1: Brier=0.2077, BSS=-1.7362, n=150
