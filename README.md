# AutoResearch Paper

Autonomous agent framework that implements a research paper from PDF to working code.

## Inputs

- Paper (PDF + supplementary material)
- Dataset (pre-downloaded)
- Optimization parameters and goals

## Pipeline

| Phase | Name | Description |
|-------|------|-------------|
| 0 | **Paper Extraction** | Extracts figures, equations, algorithms, hyperparameters, tables, and metrics from the PDF into structured files (`paper/images/`, `paper/carplanner_equations.md`, `paper/algorithms.md`, `paper/hyperparameters.md`, `paper/tables.md`, `paper/metrics.md`). Purely mechanical — no code written. |
| 1 | **Paper Contract** | Agent reads everything and writes `paper_contract.md` — the ground-truth spec: exact tensor shapes, loss terms with coefficients, training hyperparameters, evaluation protocol, ambiguities classified by impact. Also writes `progress.md` (submodule checklist). No model code yet. |
| 2 | **Data Proof** | Loads one real sample, prints shapes/dtypes/value ranges, verifies they match the paper contract, profiles loader speed. Writes `data_report.md`. Catches data pipeline problems before any model is built. |
| 3 | **Component Implementation** | Builds submodules one by one in dependency order (defined in `submodules.md`). Each gets its own file + test file. Agent doesn't move to the next submodule until the current one passes all verification gates. |
| 4 | **Integration Proof** | Wires all submodules into `train.py`. Runs a single forward+backward pass, then 10 steps, then overfits 2 samples. Proves the full pipeline works before spending GPU hours. |
| 5 | **Benchmarking** | Runs actual training at smoke then full config. Logs metrics to `results.tsv`. Checks ablation deltas match the paper's direction. |
| 6 | **Improvement** | Iterates experiments to close the gap to the paper's reported numbers. One hypothesis, one change per experiment. All history preserved. |

## Key Files

| File | Purpose |
|------|---------|
| `run_agent.py` | Agent runner — loads system prompt, tools, and LLM config |
| `program.md` | Phase-by-phase instructions the agent follows |
| `submodules.md` | Build order for all submodules with contracts and gates |
| `requirements.md` | Paper specification, dataset paths, evaluation metrics |
| `failure_patterns.md` | Known failure modes to avoid |
| `paper_contract.md` | Agent-written spec (output of Phase 1) |
| `progress.md` | Submodule checklist tracking completion |
| `results.tsv` | Experiment results log |
| `implementation/` | All generated code lives here |

## Usage

```bash
# Run from Phase N onwards
python run_agent.py --phase <N> --max-turns <T>

# Example: run Phase 2 (Data Proof) with 20 turns
python run_agent.py --phase 2 --max-turns 20
```

## Environment

Single Python venv with both torch and nuplan-devkit:

```
/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python
```
