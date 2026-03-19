# IN PROGRESS

Implement a given research paper from pdf to code. 

# Provided
- Paper
- Dataset
- optimisation parameters- goals

## Intelligence Setup

This repo now includes a repo-level intelligence configuration in
`intelligence_config.py`.

1. Copy `.env.example` to `.env`
2. Set `OPENAI_API_KEY`
3. Adjust `AUTORESEARCH_INTELLIGENCE_MODEL` if needed

Default settings:
- `AUTORESEARCH_INTELLIGENCE_PROVIDER=openai`
- `AUTORESEARCH_INTELLIGENCE_MODEL=gpt-5.4`

Quick check:

```bash
python intelligence_config.py
```

The check prints the selected provider/model and whether an API key is present,
without printing the key itself.


┌─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│        Phase        │                                                    What it does                                                    │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 0 — Paper           │ Pulls figures, equations, and metrics out of the PDF into files (paper/images/, paper/equations.md,                │
│ Extraction          │ paper/metrics.md). Purely mechanical — no code written.                                                            │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                     │ Agent reads everything and writes paper_contract.md — the ground truth spec: exact tensor shapes, loss terms with  │
│ 1 — Paper Contract  │ coefficients, training hyperparameters, evaluation protocol, ambiguities classified by impact. Also writes         │
│                     │ progress.md (the submodule checklist). No model code yet.                                                          │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2 — Data Proof      │ Loads one real nuPlan sample, prints shapes/dtypes/value ranges, verifies they match the paper contract, profiles  │
│                     │ the loader speed. Writes data_report.md. Catches data pipeline problems before any model is built.                 │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3 — Component       │ Builds the 9 submodules one by one in dependency order (as defined in submodules.md). Each gets its own file +     │
│ Implementation      │ test file. Agent doesn't move to the next submodule until the current one passes all gates.                        │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4 — Integration     │ Wires all submodules into train.py. Runs a single forward+backward pass, then 10 steps, then overfits 2 samples.   │
│ Proof               │ Proves the full pipeline works before spending GPU hours.                                                          │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5 — Benchmarking    │ Runs actual training at smoke then full config. Logs metrics to results.tsv. Checks ablation deltas match the      │
│                     │ paper's direction.                                                                                                 │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 6 — Improvement     │ Iterates experiments to close the gap to the paper's reported numbers. One hypothesis, one change per experiment.  │
│                     │ All history preserved.                                                                                             │
└─────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
