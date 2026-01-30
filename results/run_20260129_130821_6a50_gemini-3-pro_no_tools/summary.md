# Benchmark Run: run_20260129_130821_6a50_gemini-3-pro_no_tools_after_fix

Started: 2026-01-29T13:08:21.527108
Completed: 2026-01-29T13:26:45.614190
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
- tools_enabled: False
- n_samples: 750

## Results

Total predictions: 750
Beats market: 296/750 (39.5%)
Ties: 24/750 (3.2%)

### gemini-3-pro (agentic)

- Samples: 750
- Brier Score: 0.2531
- Brier Skill Score: -0.6363 (beats market: NO)
- ECE: 0.1864
- Accuracy: 0.645
- F1: 0.236

### Checkpoint Breakdown

- open_plus_1: Brier=0.2405, BSS=-0.0043, n=150
- pct_25: Brier=0.2709, BSS=-0.4686, n=150
- pct_50: Brier=0.2433, BSS=-0.6185, n=150
- pct_75: Brier=0.2505, BSS=-1.0345, n=150
- close_minus_1: Brier=0.2601, BSS=-2.4266, n=150
