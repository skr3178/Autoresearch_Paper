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

#### Phase Brief — write `phase1_brief.md` before doing anything else

Before writing `paper_contract.md`, read all Phase 0 artifacts and write `phase1_brief.md` containing:

1. **Architecture summary**: In your own words, what is the paper's core mechanism? What problem does it solve and how? (3-5 sentences, no jargon — if you can't explain it simply, you don't understand it yet.)
2. **Key equations**: List every equation that will need to be implemented. For each: equation number, what it computes, which module uses it.
3. **Key figures**: For each architecture figure, describe what components are shown and how data flows between them. Reference figure numbers.
4. **Key algorithms**: For each algorithm block, list its steps and which phase/module it belongs to.
5. **Training stages**: Does the paper train components jointly or in stages? In what order? What is frozen at each stage?
6. **Evaluation protocol**: Exactly how is the reported metric computed? Best-of-K? Single sample? What benchmark split?
7. **Open questions**: What is ambiguous or underspecified in the paper? List each uncertainty and your planned default.

This brief is a **checkpoint**: it must reference specific equation numbers, figure numbers, and algorithm names from the paper — not generic prose. If you cannot fill a section with paper-specific content, re-read the relevant artifact.

**Brief gate**: `phase1_brief.md` must exist and contain at least one equation number, one figure number, and one algorithm reference before proceeding to write `paper_contract.md`.

---

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

#### Phase Brief — write `phase2_brief.md` before writing any code

Read the dataset section of the paper, `paper_contract.md`, and `paper/images/` figures that show data examples or preprocessing pipelines. Then write `phase2_brief.md` containing:

1. **Dataset identity**: Name, source, version, license. How many scenarios/samples in train/val/test splits (exact numbers from paper).
2. **Sample anatomy**: For each model input, state its name, shape, dtype, value range, coordinate frame, and units — derived from the paper, not assumed.
3. **Preprocessing steps**: Every transformation applied to raw data before it reaches the model. List them in order with the paper section or equation that specifies each.
4. **Figures referenced**: Which figures show data examples, BEV maps, or preprocessing diagrams? Describe what each shows.
5. **Known gotchas**: Any coordinate convention, normalisation, or frame-of-reference issue that could silently produce wrong data.

**Brief gate**: `phase2_brief.md` must exist and reference specific paper section numbers and at least one figure before proceeding to write any loader code.

---

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

#### Phase Brief — write `phase3_brief.md` once before starting any submodule

Before writing any implementation code, write `phase3_brief.md` containing a section for **each submodule** in `submodules.md`:

```
## Submodule N: <name>

**Paper grounding**
- Equations: <list equation numbers that this submodule implements>
- Figures: <list figure numbers and which components in those figures map to this submodule>
- Algorithm: <algorithm name/number and which steps belong to this submodule, if any>

**Input → Output** (from paper, not from submodules.md)
- Input: <name, shape, dtype, value range as stated in paper>
- Output: <name, shape, dtype, value range as stated in paper>

**Key design decisions**
- <Any ambiguity in the paper and the chosen default>
- <Any deviation from the paper and why>

**Verification plan**
- Equation oracle: which equation will be hand-verified, and how?
- Overfit target: what does "memorize 2 samples" mean for this submodule?
```

This brief is your implementation contract. If a section cannot be filled with paper-specific content (equation numbers, figure numbers), re-read the relevant artifact before proceeding.

**Brief gate**: `phase3_brief.md` must exist and contain at least one equation number or figure number per submodule before writing any submodule code.

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

#### Phase Brief — write `phase4_brief.md` before writing `train.py`

Read Figure 2 (or whichever figure shows the full system architecture), `phase3_brief.md`, and all submodule contracts. Write `phase4_brief.md` containing:

1. **Data flow diagram**: For each component in order, state:
   - Component name → output tensor name, shape, dtype
   - Which next component consumes it and under what key
   - Any shape transformation between components
2. **Loss assembly**: List every loss term, its coefficient (from paper), which component produces it, and how they are combined. Reference the paper equation number.
3. **Training stages**: If the paper trains in stages (pretrain → finetune, IL → RL), describe each stage: which components are active, which are frozen, what data is used.
4. **Figures referenced**: Which figure(s) show the full training loop? Describe any data flow arrows that are not yet implemented by a submodule.
5. **Integration risks**: Which component interfaces are most likely to have shape/dtype mismatches? Why?

**Brief gate**: `phase4_brief.md` must exist and contain the full data flow (every component → output shape) and the loss assembly equation before writing `train.py`.

---

1. Wire all components into `implementation/train.py`.
2. **Single-step test** at debug config: one forward + backward pass. Catches cross-module shape/dtype mismatches.
3. **10-step test** at smoke config: verify loss decreases. If loss is flat or increasing after 10 steps, do not proceed.
4. **Tiny overfit test** at smoke config: train on 1-2 samples for many steps. The model should memorize them. If it can't overfit a trivial dataset, the learning pipeline is broken.
5. **Reproducibility check**: run the 10-step test twice with the same seed. Results must match exactly. If they don't, there's non-determinism that will make experiments unreliable.

**Exit gate**: Loss decreases over 10 steps. Model can overfit 1-2 samples. Two identical runs produce identical results.

---

## Phase 5: Benchmarking

**Goal**: Get a working metric and compare to the paper. Prefer ablation parity over headline-number chasing.

#### Phase Brief — write `phase5_brief.md` before running any training

Read the paper's results section, `paper/metrics.md`, `paper/tables.md`, and `paper_contract.md`. Write `phase5_brief.md` containing:

1. **Target metric**: Exact name, formula, and paper value to reproduce. Which table and row?
2. **Evaluation protocol**: Step-by-step procedure to compute the metric from model outputs — best-of-K sampling count, any filtering, normalisation, GT source, benchmark split name.
3. **Baseline ladder**: List every row in the paper's main results table that you intend to reproduce, in order from simplest to hardest. For each row: configuration name, expected metric value, which components are active.
4. **Ablation table**: If the paper has an ablation, list each ablation row with expected metric delta. This is your primary correctness signal — matching ablation directions is stronger evidence than matching headline numbers.
5. **Figures referenced**: Which figures show training curves, metric vs. epoch, or qualitative results? What trends should you expect to see?
6. **Hardware adaptation**: If paper used different hardware, note any changes to batch size, learning rate scaling, or step count, and the paper section that justifies each.

**Brief gate**: `phase5_brief.md` must exist and contain the target metric value, the exact evaluation protocol steps, and the ablation table before starting any training run.

---

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

#### Phase Brief — write `phase6_brief.md` before starting the improvement loop

Read `phase5_brief.md`, `results.tsv`, and the paper's ablation section. Write `phase6_brief.md` containing:

1. **Current gap**: Actual metric achieved vs. paper target. Which ablation rows have been matched and which haven't?
2. **Gap diagnosis**: For each component of the gap, cite the paper section or figure that describes the expected behaviour, and describe how the current implementation deviates.
3. **Improvement hypotheses**: List candidate causes in order of estimated impact. For each: what paper evidence supports this hypothesis, and what single change would test it?
4. **High-impact ambiguities**: From `paper_contract.md`'s ambiguity register, which medium/high-impact items have not yet been tested? List the alternative interpretations to try.
5. **Stop criteria**: Exact numeric targets that define "close enough" — derived from `paper_contract.md` tolerance definition.

**Brief gate**: `phase6_brief.md` must exist and reference the current metric gap with specific numbers and at least two paper-grounded improvement hypotheses before starting the experiment loop.

---

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