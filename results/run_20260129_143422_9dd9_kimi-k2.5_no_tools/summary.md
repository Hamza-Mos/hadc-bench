# Benchmark Run: run_20260129_143422_9dd9_kimi-k2.5_no_tools_after_fix

Started: 2026-01-29T14:34:22.628605
Completed: 2026-01-29T14:48:54.817121
Status: completed

## Configuration

- model_ids: ['kimi-k2.5']
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
Beats market: 243/750 (32.4%)
Ties: 26/750 (3.5%)

### kimi-k2.5 (agentic)

- Samples: 750
- Brier Score: 0.2214
- Brier Skill Score: -0.4315 (beats market: NO)
- ECE: 0.1219
- Accuracy: 0.635
- F1: 0.271

### Checkpoint Breakdown

- open_plus_1: Brier=0.2374, BSS=+0.0084, n=150
- pct_25: Brier=0.2217, BSS=-0.2017, n=150
- pct_50: Brier=0.2230, BSS=-0.4833, n=150
- pct_75: Brier=0.2185, BSS=-0.7744, n=150
- close_minus_1: Brier=0.2064, BSS=-1.7189, n=150
