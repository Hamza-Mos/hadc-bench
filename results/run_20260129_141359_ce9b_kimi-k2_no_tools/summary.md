# Benchmark Run: run_20260129_141359_ce9b_kimi-k2_no_tools_after_fix

Started: 2026-01-29T14:13:59.965438
Completed: 2026-01-29T14:34:22.581079
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
- tools_enabled: False
- n_samples: 750

## Results

Total predictions: 750
Beats market: 303/750 (40.4%)
Ties: 22/750 (2.9%)

### kimi-k2 (agentic)

- Samples: 750
- Brier Score: 0.2322
- Brier Skill Score: -0.5013 (beats market: NO)
- ECE: 0.1612
- Accuracy: 0.631
- F1: 0.249

### Checkpoint Breakdown

- open_plus_1: Brier=0.2308, BSS=+0.0360, n=150
- pct_25: Brier=0.2485, BSS=-0.3470, n=150
- pct_50: Brier=0.2204, BSS=-0.4661, n=150
- pct_75: Brier=0.2174, BSS=-0.7661, n=150
- close_minus_1: Brier=0.2438, BSS=-2.2112, n=150
