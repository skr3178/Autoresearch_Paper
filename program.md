# autoresearch-paper

Autonomous agent framework for implementing research papers from scratch.

---

## Before You Start

Read these files completely before writing any code:

1. `requirements.md` — paper summary, dataset, target metric, reported baseline
2. `failure_patterns.md` — general-purpose failure catalog from past implementations
3. `paper/` — the paper PDF and any supplementary material

You build everything from scratch inside `implementation/`. There is no existing code to modify.

The implementation proceeds through seven phases. Each phase has an exit gate — do not advance until the gate is met. All experiments are preserved (never delete history).

**IMPORTANT: Do NOT run any git commands (git add, git commit, git checkout, git branch, etc.). All work is local only. Never create branches, stage files, or commit anything.**

---

## Phase 0: Paper Extraction

**Goal**: Extract all structured information from the paper PDF into machine-readable artifacts before any analysis or coding.

1. **Extract all figures** from the paper PDF. Save each as a separate image in `paper/images/` named `fig_<N>.png` (e.g. `fig_1.png`, `fig_2.png`). Include sub-figures separately if they represent distinct diagrams.

2. **Annotate each figure** with a companion text file `paper/images/fig_<N>.txt` containing:
   - The figure's caption (verbatim from the paper)
   - All textual information visible inside the figure (axis labels, legends, dimension annotations, layer names, arrow labels, etc.)
   - Pertinent text from the paper body that references or describes the figure — architectural details, dimensions, data flow, design rationale. Cite the section and paragraph.

3. **Extract all equations** into `paper/carplanner_equations.md`. For each equation:
   - The equation number from the paper
   - The equation itself (in LaTeX or Unicode math)
   - The surrounding context: what it computes, which variables are inputs/outputs, any constraints or conditions stated in the text

4. **Extract referenced metrics** into `paper/metrics.md`. For each metric reported in the paper:
   - Table number and row/column
   - Metric name, value, and which model/ablation it corresponds to
   - Any conditions (dataset split, best-of-K, thresholds)

5. **Extract all algorithms/pseudocode** into `paper/algorithms.md`. For each algorithm block:
   - Algorithm number and name
   - Full pseudocode verbatim
   - Inputs, outputs, and any hyperparameters referenced
   - Which section it appears in

6. **Extract all hyperparameters** into `paper/hyperparameters.md`. For every hyperparameter mentioned anywhere in the paper:
   - Name and symbol
   - Value
   - Where it is used (which module/loss/training stage)
   - Section/table it appears in

7. **Extract all tables** into `paper/tables.md`. For every table in the paper:
   - Table number and caption
   - Full table contents (rows, columns, values)
   - What the table demonstrates (ablation, comparison, analysis)
   - Which rows/configurations are the paper's recommended defaults

**Exit gate**: Every figure (annotated), equation, metric, algorithm, hyperparameter, and table from the paper is extracted. These artifacts are the reference source for all subsequent phases.

---

## Phase 1: Paper Contract

**Goal**: Prove you understand what you're implementing before writing any code.

1. **Read everything**: `requirements.md`, `failure_patterns.md`, paper in `paper/`, and the extracted artifacts from Phase 0 (`paper/images/`, `paper/carplanner_equations.md`, `paper/metrics.md`).
2. **Write `paper_contract.md`** containing:
   - **Dataset contract**: name, source, format of one sample (type, shape, dtype, value range), train/val/test split sizes, coordinate frame and units (if spatial data), any required preprocessing
   - **Architecture contract**: every module the paper defines, with input/output shapes in paper notation
   - **Training contract**: optimizer, LR schedule, batch size, number of steps/epochs, loss function with every term and coefficient mapped: `paper symbol → semantic meaning → value`
   - **Evaluation contract**: exact metric name, computation procedure (best-of-K or single-sample, thresholds, normalization, GT source, sampling count, seed policy, any paper-specific filtering), how to compare to paper baseline (metric direction, expected value, tolerance given hardware/data differences)
   - **Inference contract**: how to go from trained model to evaluation output (sampling procedure, post-processing, any schedule differences from training)
   - **Ambiguity register**: every detail the paper leaves unclear, classified by impact:
     - **Low**: unlikely to affect metric (e.g. weight init variant). Use simplest default.
     - **Medium**: may affect metric. Make configurable, note the default and rationale.
     - **High**: likely affects metric significantly. Implement at least two candidate interpretations in config. Justify the chosen default. Flag for revisiting if metric doesn't match.
3. **Decompose into components** (see "How to Decompose").
4. **Write `progress.md`**: ordered component checklist with exit criteria per component.
5. **Initialize `results.tsv`** with the header row.

**Exit gate**: `paper_contract.md` and `progress.md` are written. Every loss term has a mapping. Every ambiguity is classified.

---

## How to Decompose a Paper

**Do NOT decompose by paper sections.** Papers are structured for readability, not implementability. Instead:

### Step 1: Identify the core learning component
Every paper has one central mechanism. Find it. Build and verify it first.

### Step 2: Order by dependency and risk
Build dependencies before dependents. Build risky/uncertain components before safe ones. If the core mechanism doesn't work, nothing else matters.

### Step 3: Start with the simplest ablation
If the paper has an ablation table, implement the simplest row first (fewest components, single modality, no auxiliary losses). The ablation table tells you which components contribute most — add those in order of impact.

### Step 4: Plan for staged training
If the paper has multiple components trained jointly, plan to train them separately first. Joint training often fails due to gradient conflicts. Common pattern: train A → freeze A → train B on A's outputs. Only attempt joint training after both work independently.

### Step 5: Multiple config tiers
Define configs for different scales:
- **debug**: smallest possible dimensions (2 layers, 64 hidden, batch 2, 10 steps). Runs in <5 seconds.
- **smoke**: small but realistic (fewer layers, small batch). Runs in 1-2 minutes.
- **full**: paper-scale or hardware-adapted. Used for real training runs.

---

## Phase 2: Data Proof

**Goal**: Prove the data pipeline is correct before writing any model code.

1. **Load one sample** from disk. Print its type, shape, dtype, value range.
2. **Visualize it** — render/plot the sample and verify it looks correct (images look like images, coordinates are in the right frame, sequences have expected length).
3. **Verify shapes** match what `paper_contract.md` specifies.
4. **Verify units and coordinates** — if spatial data, confirm the coordinate convention matches the paper. Test with a known point if possible.
5. **Verify split counts** — number of train/val/test samples matches the paper or `requirements.md`.
6. **Profile the loader**: load 10 batches at batch_size=2, time it. If >2 seconds, vectorize before proceeding.
7. **Write `data_report.md`** documenting all findings: sample structure, shapes, dtype, value ranges, split sizes, loader throughput, any discrepancies from the paper.

**Exit gate** — ALL of the following must pass before moving to Phase 3:
1. `implementation/data_proof.py` runs to completion with exit_code=0 (no exceptions, full output printed)
2. `implementation/data_loader.py` loads at least one batch without error — run: `python implementation/data_loader.py` and verify exit_code=0
3. `data_report.md` exists and contains actual printed values: scenario count, ego state shapes/dtypes, tracked object count, loader throughput in seconds
4. The shapes and dtypes in `data_report.md` match what `paper_contract.md` specifies

Do not declare Phase 2 complete until you have run both scripts and confirmed exit_code=0 for each.

---

## Phase 3: Component Implementation

**Read `submodules.md` first.** It defines the exact build order for all submodules with input/output contracts and hard verification gates. Work through it strictly in order — **never start submodule N+1 until submodule N's gate is fully met.**

After each submodule gate passes:
- Mark the submodule checkbox `✅` in `progress.md`

### For each submodule:

#### Step 1: Understand before coding — MANDATORY paper artifact reading (ONCE per session)

**Check first**: Does `paper_context.md` exist? If yes, skip to Step 2 — all paper artifacts are already loaded for this session.

If `paper_context.md` does NOT exist, read ALL of the following exactly once:

1. **Equations**: `read_file("paper/carplanner_equations.md")`
2. **Algorithms**: `read_file("paper/algorithms.md")`
3. **Hyperparameters**: `read_file("paper/hyperparameters.md")`
4. **Tables**: `read_file("paper/tables.md")`
5. **Paper contract**: `read_file("paper_contract.md")`
6. **Failure patterns**: `read_file("failure_patterns.md")`
7. **ALL paper figures**: for every `.png` in `paper/images/`, call `read_image`. Also read each companion `.txt`.

After reading all of the above, write `paper_context.md` with this exact content:
```
# Paper Context
status: loaded
figures: <comma-separated list of figure filenames you read>
```

This file is your signal that paper artifacts are loaded. Never re-read paper artifacts in the same session if this file exists.

#### Step 2: Implement minimally
- Create the files listed in `submodules.md` for this submodule.
- Parameterize all dimensions from a config dict — never hardcode spatial sizes, channel counts, sequence lengths, or vocabulary sizes.
- Implement only what is needed for this submodule's gate. Do not pre-implement the next submodule.

#### Step 3: Verify (not just test)

Write `implementation/test_<submodule>.py`. The tests must prove correctness, not just executability. Run every test listed in this submodule's **Verification gate** in `submodules.md`, plus:

**Executability tests:**
1. Shape assertions — verify every tensor shape matches paper contract and `submodules.md` spec.
2. Dtype assertions — verify no silent type promotions.
3. Forward + backward pass at debug config.

**Correctness tests:**
4. Equation oracle test — for at least one equation, hand-compute the expected output for a tiny input (2-3 elements) and assert the code produces the same result within tolerance.
5. Tiny overfit test — can the submodule memorize 1-2 samples? If not, something fundamental is wrong.
6. Output diversity — feed two different inputs, verify outputs differ (cosine similarity <0.95).
7. Sign/direction test — where applicable, verify masks, schedules, or gradients have the correct sign/direction.
8. Invariance/equivariance test — if the submodule should be invariant or equivariant to something (permutation, rotation, translation), test it.

Run: `uv run implementation/test_<submodule>.py`

#### Step 4: Document proof
After tests pass, update `proof.md` for this submodule:
- What assumptions were made and why
- What evidence confirms correctness (which tests pass, what they verify)
- Any remaining uncertainty

#### Step 5: Figure verification — MANDATORY before marking ✅

Before marking any submodule complete, you MUST do a visual architecture verification:

1. Use `list_dir("paper/images")` to see all available figures and annotation files.
2. Call `read_image` on every figure in `paper/images/`. Also read every companion `.txt` annotation file.
3. **Classify each figure** independently before doing anything else:
   - **Architecture figure**: shows components, blocks, data flow arrows, module connections, or algorithm steps → must be compared against code
   - **Results figure**: shows metric numbers, performance tables, training curves, qualitative trajectory plots, or ablation results → cannot be verified without a trained model; note it in `proof.md` as an evaluation target for a later phase and skip code comparison
   Only architecture figures proceed to the mapping step below.
4. **Build an explicit figure-to-code mapping** for this submodule using only architecture figures. For every visual component in those figures (blocks, arrows, modules, data flows), write a mapping entry:
   ```
   Figure N, component "<name>" → <file.py>:<ClassName or function> (or ⚠️ NOT FOUND)
   ```
   Then for every class/function in the implementation, write:
   ```
   <file.py>:<ClassName> → Figure N, component "<name>" (or ⚠️ NO FIGURE ANCHOR)
   ```
5. Save this mapping to `proof.md` under this submodule's section.
6. For every `⚠️ NOT FOUND` entry — a figure component with no code: implement it.
7. For every `⚠️ NO FIGURE ANCHOR` entry — code with no figure: verify it is justified by an equation or algorithm, or remove it if it is an invention not grounded in the paper.
8. Fix all discrepancies, re-run tests, verify they still pass.

Only after completing this mapping and resolving all `⚠️` entries may you mark `✅` in `progress.md`.

#### Step 6: Gate check
- **PASS**: All code tests pass AND figure verification is complete and documented in `proof.md`. Then:
  - Mark `✅` in `progress.md`
  - Only now, proceed to the next submodule in `submodules.md`.
- **FAIL**: Debug (max 3 attempts). If stuck, mark `⚠️ BLOCKED` in `progress.md` with notes and the specific failing gate check. Do not skip to the next submodule.

---

## Phase 4: Integration Proof

**Goal**: Prove the full pipeline works end-to-end before investing GPU hours.

1. Wire all components into `implementation/train.py`.
2. **Single-step test** at debug config: one forward + backward pass. Catches cross-module shape/dtype mismatches.
3. **10-step test** at smoke config: verify loss decreases. If loss is flat or increasing after 10 steps, do not proceed.
4. **Tiny overfit test** at smoke config: train on 1-2 samples for many steps. The model should memorize them. If it can't overfit a trivial dataset, the learning pipeline is broken.
5. **Reproducibility check**: run the 10-step test twice with the same seed. Results must match exactly. If they don't, there's non-determinism that will make experiments unreliable.

**Exit gate**: Loss decreases over 10 steps. Model can overfit 1-2 samples. Two identical runs produce identical results.

---

## Phase 5: Benchmarking

**Goal**: Get a working metric and compare to the paper. Prefer ablation parity over headline-number chasing.

### First run: smoke config
```
uv run implementation/train.py --config smoke > run.log 2>&1
```
Verify loss decreases, no NaN, memory is reasonable.

### Second run: full config
```
uv run implementation/train.py --config full > run.log 2>&1
```
Parse metric (grep pattern from `requirements.md`), log to `results.tsv`.

### Reproducibility gate
Run full config twice. Results must be within tolerance (define tolerance in `paper_contract.md`). If not, the result is exploratory — not evidence.

### Ablation parity check
If the paper has an ablation table:
1. Run the simplest ablation first (baseline).
2. Add one component at a time.
3. Verify each addition moves the metric in the **same direction** as the paper reports.

Reproducing the ablation deltas is stronger evidence of a correct implementation than matching a headline score.

### What to log during training
Log every N steps (N ≈ 100):
- Each loss component separately — never just total loss.
- Gradient norm per parameter group.
- Any component-specific health metric.

### Fast-fail checks
```
grep "^metric:\|^loss:\|^ERROR\|Traceback\|NaN\|nan" run.log | head -20
```
If crashed: `tail -n 80 run.log` → check `failure_patterns.md` → fix root cause.

**Exit gate**: At least one full run completes with a real metric value. Reproducibility verified. If paper has ablations, at least the baseline ablation reproduces.

---

## Phase 6: Improvement

Once benchmarking produces a working baseline, improve toward the paper's reported result.

Each experiment:

1. Compare metric to paper baseline (check direction from `requirements.md`).
2. Diagnose the gap. Use logged diagnostics, not guesswork. Check:
   - Do loss components match expected magnitudes from the paper?
   - Are intermediate representations healthy (reasonable mean/std, diverse across inputs)?
   - Does the evaluation protocol exactly match `paper_contract.md`?
3. Hypothesize ONE cause. Make ONE change. **Simplicity criterion**: a tiny improvement that adds complexity is not worth it. Removing code for equal or better results is always a win.
4. `git commit` (keep all experiments — see "Experiment History" below).
5. Run training: `uv run implementation/train.py --config full > run.log 2>&1`
6. Parse metric, log to `results.tsv` and `decision_log.md`.
7. If metric improved → mark `keep` in results.tsv.
8. If metric worsened or equal → mark `discard` in results.tsv. Do NOT delete the commit — `git revert HEAD` if you want to undo. All history is preserved.

**Timeout**: 2× the `training_budget` in `requirements.md`. Exceeding = crash.

**Autonomous operation**: Do not pause to ask the human. Loop until manually interrupted.

**Stop conditions** (any of these ends the loop):
- Ablation parity achieved — each component's contribution matches the paper's direction and approximate magnitude.
- Metric within tolerance of paper baseline (tolerance defined in `paper_contract.md`).
- Remaining gap is isolated and documented — you know exactly which component/ambiguity is responsible but cannot resolve it without human input.

**Stuck after 5+ failed experiments:**
- Re-read the paper with your code open. Map each equation to its code line.
- Re-read `failure_patterns.md`.
- Revisit high-impact ambiguities in `paper_contract.md` — try the alternative interpretation.
- Strip back to last known-good state and verify it still works.
- Check the evaluation contract — are you computing the metric exactly as the paper does?

---

## Experiment History

**Never delete experiment history.** Every commit is data.

`results.tsv` — tab-separated:
```
commit	metric	memory_gb	status	description
```
- `status`: `keep`, `discard`, or `crash`
- All experiments logged, including failures.

`decision_log.md` — append-only log of reasoning:
```
## Experiment: <short hash>
**Hypothesis**: <what you expected and why>
**Change**: <what you changed>
**Result**: <metric value, keep/discard>
**Diagnosis**: <why it worked/failed, what you learned>
```

This log prevents repeating failed experiments and helps diagnose systematic issues.

---

## Principles

Hard-won lessons from implementing multiple papers.

### 1. Incremental, not big-bang
Build the core, test it, then add one thing at a time. Each addition gets its own test. If adding a feature makes things worse, you know exactly what broke.

### 2. Paper ≠ implementation
The paper is idealized. Your implementation must handle hardware constraints, training stability, data pipeline performance, and numerical precision. Adapt early.

### 3. Monitor components, not just totals
Log every loss term separately. Log gradient norms. When something goes wrong, you need to see WHERE.

### 4. Verify math line-by-line
Maintain `notation_map.md`: paper equation → code line. Loss coefficient inversions, wrong activations, and missing algorithm steps are the most common and most devastating bugs. They're invisible in shape tests.

### 5. Separate training stages when joint fails
Default to staged training. Joint training only after components work independently.

### 6. Cache frozen computations
Precompute frozen component outputs to disk. Never re-run a frozen component in the training loop. 10-100× speedup.

### 7. Test with simple data first
Validate the pipeline on simple/synthetic data before real data. Simple data lets you verify correctness by eye.

### 8. Make ambiguities configurable
Don't hardcode guesses. Make them config parameters with comments citing the paper section.

### 9. Ablation parity over headline numbers
Reproducing the ablation table (each component's delta) is stronger evidence than matching a raw score that could be right for the wrong reasons.

### 10. Profile before optimizing
Measure where time is spent. Data loader bottlenecks matter more than model optimizations.

### 11. Preserve all history
Every experiment is data. Never `git reset --hard`. Use `git revert` if you need to undo. Log everything in `results.tsv` and `decision_log.md`.

### 12. Training completed ≠ correct
A run that finishes without crashing proves nothing about correctness. Verify outputs, check ablation deltas, test evaluation protocol, compare intermediate quantities.





to the instruction set for the paper implementor,we added these specific tests: 1. Figure extractor from the paper. all the figures should be clipped and stored in a
seperate folder called images. 2. Each figure should be accompanied by a text file which includes the caption for the figure, extract textual information in the    
figure, and extracts pertaining text from the paper pertaining the figure- say architectural details, dimensions etc. 3. all equation listed should eb extracted    
from the figure into a file called equations. 4. referenced test metrics should be extracted too.