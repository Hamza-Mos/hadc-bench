# Benchmark Run: run_20260129_144854_49bd_trinity-large-preview_no_tools_after_fix

Started: 2026-01-29T14:48:54.871180
Completed: 2026-01-29T14:50:30.493934
Status: completed

## Configuration

- model_ids: ['trinity-large-preview']
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
Beats market: 225/750 (30.0%)
Ties: 10/750 (1.3%)

### trinity-large-preview (agentic)

- Samples: 750
- Brier Score: 0.2684
- Brier Skill Score: -0.7354 (beats market: NO)
- ECE: 0.2027
- Accuracy: 0.563
- F1: 0.317

### Checkpoint Breakdown

- open_plus_1: Brier=0.2702, BSS=-0.1286, n=150
- pct_25: Brier=0.2625, BSS=-0.4228, n=150
- pct_50: Brier=0.2962, BSS=-0.9705, n=150
- pct_75: Brier=0.2449, BSS=-0.9889, n=150
- close_minus_1: Brier=0.2681, BSS=-2.5319, n=150
