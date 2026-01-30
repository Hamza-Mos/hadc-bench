# Benchmark Run: run_20260129_113319_e29c_gpt-5.2-xhigh_no_tools_after_fix

Started: 2026-01-29T11:33:19.830574
Completed: 2026-01-29T11:35:04.739513
Status: completed

## Configuration

- model_ids: ['gpt-5.2-xhigh']
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
Beats market: 269/750 (35.9%)
Ties: 20/750 (2.7%)

### gpt-5.2-xhigh (agentic)

- Samples: 750
- Brier Score: 0.2153
- Brier Skill Score: -0.3921 (beats market: NO)
- ECE: 0.1140
- Accuracy: 0.640
- F1: 0.274

### Checkpoint Breakdown

- open_plus_1: Brier=0.2095, BSS=+0.1252, n=150
- pct_25: Brier=0.2218, BSS=-0.2024, n=150
- pct_50: Brier=0.2215, BSS=-0.4736, n=150
- pct_75: Brier=0.2071, BSS=-0.6817, n=150
- close_minus_1: Brier=0.2166, BSS=-1.8532, n=150
