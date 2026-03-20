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

4. **Extract all algorithms/pseudocode** into `paper/algorithms.md`. For each algorithm block:
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

**Exit gate**: Every figure (annotated), equation, metric, algorithm, hyperparameter, and table from the paper is extracted. Write `phase0_report.md` listing each extracted artifact (filename and item count). Do not proceed until `phase0_report.md` exists.

---

## Phase 1: Paper Contract

**Goal**: Prove you understand what you're implementing before writing any code.

#### Phase Brief — write `phase1_brief.md` before doing anything else

Read all Phase 0 artifacts (`paper/carplanner_equations.md`, `paper/algorithms.md`, `paper/hyperparameters.md`, `paper/tables.md`, all figures in `paper/images/`) and write `phase1_brief.md` containing:

1. **Architecture summary**: In your own words, what is the paper's core mechanism? What problem does it solve and how? (3-5 sentences, no jargon — if you can't explain it simply, you don't understand it yet.)
2. **Key equations**: List every equation that will need to be implemented. For each: equation number, what it computes, which module uses it.
3. **Key figures**: For each architecture figure, describe what components are shown and how data flows between them. Reference figure numbers.
4. **Key algorithms**: For each algorithm block, list its steps and which phase/module it belongs to.
5. **Training stages**: Does the paper train components jointly or in stages? In what order? What is frozen at each stage?
6. **Evaluation protocol**: Exactly how is the reported metric computed? Best-of-K? Single sample? What benchmark split?
7. **Open questions**: What is ambiguous or underspecified in the paper? List each uncertainty and your planned default.

**Forbidden entries**: "equation TBD", "figure to be checked", "algorithm unclear" — use `query_pdf` and fill in the actual content.

**Brief gate**: `phase1_brief.md` must exist and contain at least one equation number, one figure number, and one algorithm reference before proceeding.

---

1. **Read everything**: `requirements.md`, `failure_patterns.md`, paper in `paper/`, and all Phase 0 artifacts.
2. **Write `phase1_report.md`** containing:
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

**Exit gate**: `phase1_report.md` and `progress.md` are written. Every loss term has a paper-sourced coefficient. Every ambiguity is classified. Do not proceed until both files exist.

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

Read the dataset section of the paper, `phase1_report.md`, and `paper/images/` figures that show data examples or preprocessing pipelines. Then write `phase2_brief.md` containing:

1. **Dataset identity**: Name, source, version, license. How many scenarios/samples in train/val/test splits (exact numbers from paper).
2. **Sample anatomy**: For each model input, state its name, shape, dtype, value range, coordinate frame, and units — derived from the paper, not assumed.
3. **Preprocessing steps**: Every transformation applied to raw data before it reaches the model. List them in order with the paper section or equation that specifies each.
4. **Figures referenced**: Which figures show data examples, BEV maps, or preprocessing diagrams? Describe what each shows.
5. **Known gotchas**: Any coordinate convention, normalisation, or frame-of-reference issue that could silently produce wrong data.

**Forbidden entries in `phase2_brief.md`**: "format TBD", "shape unclear", "to be verified from data" are gate failures — query the PDF and fill in the actual content before writing the brief.

**Brief gate**: `phase2_brief.md` must exist and reference specific paper section numbers and at least one figure before proceeding to write any loader code.

---

1. **Load one sample** from disk. Print its type, shape, dtype, value range.
2. **Visualize it** — render/plot the sample and verify it looks correct (images look like images, coordinates are in the right frame, sequences have expected length).
3. **Verify shapes** match what `phase1_report.md` specifies.
4. **Verify units and coordinates** — if spatial data, confirm the coordinate convention matches the paper. Test with a known point if possible.
5. **Verify split counts** — number of train/val/test samples matches the paper or `requirements.md`.
6. **Profile the loader**: load 10 batches at batch_size=2, time it. If >2 seconds, vectorize before proceeding.
7. **Write `phase2_report.md`** documenting all findings: sample structure, shapes, dtype, value ranges, split sizes, loader throughput, any discrepancies from the paper.

**Exit gate** — ALL of the following must pass before moving to Phase 3:
1. `implementation/data_proof.py` runs to completion with exit_code=0 (no exceptions, full output printed)
2. `implementation/data_loader.py` loads at least one batch without error — run: `python implementation/data_loader.py` and verify exit_code=0
3. `data_report.md` exists and contains actual printed values: scenario count, ego state shapes/dtypes, tracked object count, loader throughput in seconds
4. The shapes and dtypes in `data_report.md` match what `phase1_report.md` specifies

Do not declare Phase 2 complete until you have run both scripts and confirmed exit_code=0 for each.

---

## Phase 3: Component Implementation

**Read `submodules.md` first.** It defines the exact build order for all submodules with input/output contracts and hard verification gates. Work through it strictly in order — **never start submodule N+1 until submodule N's gate is fully met.**

After each submodule gate passes:
- Mark the submodule checkbox `✅` in `progress.md`

#### Phase Brief — write `phase3_brief.md` once before starting any submodule

**Before writing the brief**, do ALL of the following in order:
1. Read `paper/carplanner_equations.md`, `paper/algorithms.md`, `paper/hyperparameters.md`, `paper/tables.md`.
2. Read every figure in `paper/images/` and its companion `.txt`.
3. For each submodule in `submodules.md`, call `query_pdf` with a targeted question: *"What equations, figures, algorithms, and hyperparameters in the paper correspond to [submodule name]? Quote the relevant text."* Record the answer before writing the brief section.

Only after completing these three steps, write `phase3_brief.md` with a section for **each submodule**:

```
## Submodule N: <name>

**Paper grounding** (must be filled from actual paper content — no placeholders)
- Equations: <exact equation numbers, e.g. "Eq 3, Eq 5"> — if none exist in the paper write "PDF query confirms: absent"
- Figures: <exact figure numbers and component names visible in the figure, e.g. "Fig 2: Mode Selector block"> — if none write "PDF query confirms: absent"
- Algorithm: <exact algorithm number and step range, e.g. "Algorithm 1, Steps 4-7"> — if none write "PDF query confirms: absent"

**Input → Output** (from paper, not from submodules.md — quote the paper section)
- Input: <name, shape, dtype, value range — cite paper section>
- Output: <name, shape, dtype, value range — cite paper section>

**Key design decisions**
- For each ambiguity: state exactly what the paper says, what is unclear, and the chosen default
- For any detail NOT in the paper: write "NOT IN PAPER — will not implement" (do not invent)

**Verification plan**
- Equation oracle: which exact equation (by number) will be hand-verified, and how?
- Overfit target: what does "memorize 2 samples" mean for this submodule?
```

**Forbidden entries** — the brief gate will REJECT any of the following:
- `Equations: N/A` — use "PDF query confirms: absent" instead (proves you checked)
- `Equations: <formula> (if specified)` — check the paper; do not write conditional placeholders
- `Figures: Fig(s) showing... (to be mapped after...)` — map figures now, before writing the brief
- Any design decision that invents a mechanism not in the paper (e.g. "implement CNN + FiLM", "implement squared jerk penalty") — write "NOT IN PAPER — will not implement" instead

**Brief gate**: `phase3_brief.md` must exist and for EVERY submodule either (a) cite a real equation number and figure number from the paper, or (b) contain "PDF query confirms: absent" for any section that is genuinely absent from the paper. Placeholders, conditionals, and invented mechanisms are gate failures.

### For each submodule:

#### Step 1: Understand before coding — MANDATORY paper artifact reading (ONCE per session)

**Check first**: Does `paper_context.md` exist? If yes, skip to Step 2 — all paper artifacts are already loaded for this session.

If `paper_context.md` does NOT exist, read ALL of the following exactly once:

1. **Equations**: `read_file("paper/carplanner_equations.md")`
2. **Algorithms**: `read_file("paper/algorithms.md")`
3. **Hyperparameters**: `read_file("paper/hyperparameters.md")`
4. **Tables**: `read_file("paper/tables.md")`
5. **Paper contract**: `read_file("phase1_report.md")`
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
- **Every non-trivial computation must cite its paper source in a comment**:
  - If it implements a paper equation: `# Eq N: <formula> (paper/carplanner_equations.md)`
  - If it follows a paper algorithm step: `# Algorithm N, Step M (paper/algorithms.md)`
  - If it uses a paper hyperparameter: `# <symbol>=<value> from Table N / Section X (paper/hyperparameters.md)`
  - If NO paper source exists for a computation after checking the extracted artifacts: call `query_pdf` with a targeted question (e.g. "Is there an equation or description for <computation>?"). If the PDF confirms it is absent, then and only then write: `# ⚠️ INVENTION: not in paper (confirmed by PDF query) — justified by <reason>`
- **Acceptable inventions** (structural scaffolding that any PyTorch model needs): layer normalization, dropout, activation functions (ReLU, GELU, SiLU), weight initialization, gradient clipping, learning rate warm-up, padding/masking boilerplate. These do NOT require `⚠️ INVENTION` comments.
- **Unacceptable inventions** (must NOT be added without paper grounding): loss terms, reward signals, penalty functions, auxiliary objectives, attention mechanisms, temperature or scaling factors applied to loss or logits, special handling for edge cases (NaN, collision, clip values). If the paper does not define it, do not implement it — remove it instead.

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
After tests pass, update `proof.md` for this submodule with ALL of the following sections:

**4a. Equation citations**
For every formula in the implementation, write a mapping entry:
```
Eq N (<formula>) → <file.py>:<function or line> — MATCH / MISMATCH: <notes>
```
For every computation with NO paper equation:
```
⚠️ INVENTION: <what it does> — justified by <reason>, or REMOVED if unjustified
```
Cross-check against `paper/carplanner_equations.md`. If an equation in the paper is NOT implemented, note it as a gap.

**4b. Algorithm citations**
For each step in `paper/algorithms.md` that belongs to this submodule:
```
Algorithm N, Step M: <step description> → <file.py>:<function> — IMPLEMENTED / MISSING
```

**4c. Hyperparameter citations**
For every numeric constant or coefficient in the implementation:
```
<variable_name> = <value> → paper source: <symbol>, Table N / Section X (paper/hyperparameters.md) — MATCH / MISMATCH / INVENTED
```

**4d. Correctness evidence**
- What assumptions were made and why
- Which tests pass and what they verify
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
ALL of the following must be true before marking ✅:

1. All code tests pass (shapes, dtypes, forward/backward, equation oracle, overfit)
2. Figure verification complete in `proof.md` — forward + reverse mappings, no unresolved ⚠️ entries
3. **Equation citations complete in `proof.md`** — every formula maps to an equation in `paper/carplanner_equations.md`, OR is explicitly declared as `⚠️ INVENTION` with justification
4. **Algorithm citations complete in `proof.md`** — every algorithm step from `paper/algorithms.md` that belongs to this submodule is either IMPLEMENTED or noted as a deliberate gap
5. **Hyperparameter citations complete in `proof.md`** — every numeric constant traces to `paper/hyperparameters.md` or is declared as INVENTED
6. **No silent inventions** — any computation not grounded in the paper must be explicitly flagged. Undeclared inventions are treated as gate failures.

If ALL pass:
- Mark `✅` in `progress.md`
- Only now, proceed to the next submodule in `submodules.md`.

If ANY fail:
- Debug (max 3 attempts). If stuck, mark `⚠️ BLOCKED` in `progress.md` with notes. Do not skip to the next submodule.

---

## Phase 4: Integration Proof

**Goal**: Prove the full pipeline works end-to-end before investing GPU hours.

#### Phase Brief — write `phase4_brief.md` before writing `implementation/train.py`

Read Figure 2 (or whichever figure shows the full system architecture), `phase3_brief.md`, and all submodule contracts. Write `phase4_brief.md` containing:

1. **Data flow diagram**: For each component in order, state:
   - Component name → output tensor name, shape, dtype
   - Which next component consumes it and under what key
   - Any shape transformation between components
2. **Loss assembly**: List every loss term, its coefficient (from paper), which component produces it, and how they are combined. Reference the paper equation number.
3. **Training stages**: If the paper trains in stages (pretrain → finetune, IL → RL), describe each stage: which components are active, which are frozen, what data is used.
4. **Figures referenced**: Which figure(s) show the full training loop? Describe any data flow arrows that are not yet implemented by a submodule.
5. **Integration risks**: Which component interfaces are most likely to have shape/dtype mismatches? Why?

**Forbidden entries in `phase4_brief.md`**:
- Equation numbers written as placeholders (e.g. "Eq (8)-(10) placeholders", "equation TBD") — query the PDF and cite the real equation number
- Components in the data flow that are not grounded in the paper (e.g. ConsistencyModule if absent from paper) — only include components confirmed by `query_pdf`
- Loss terms with coefficient "TBD" or "to be confirmed" — find the value in `paper/hyperparameters.md` or via `query_pdf` before writing the brief
- Training stage descriptions like "frozen for now" or "not yet integrated" — describe exactly what the paper says about training stages

**Brief gate**: `phase4_brief.md` must exist and contain the full data flow (every component → output shape, each citing a paper figure or section), the complete loss assembly with real equation numbers and coefficient values, and the training stage description — all confirmed from the paper. No placeholders.

---

1. Wire all components into `implementation/train.py`.
2. **Single-step test** at debug config: one forward + backward pass. Catches cross-module shape/dtype mismatches.
3. **10-step test** at smoke config: verify loss decreases. If loss is flat or increasing after 10 steps, do not proceed.
4. **Tiny overfit test** at smoke config: train on 1-2 samples for many steps. The model should memorize them. If it can't overfit a trivial dataset, the learning pipeline is broken.
5. **Reproducibility check**: run the 10-step test twice with the same seed. Results must match exactly. If they don't, there's non-determinism that will make experiments unreliable.

**Exit gate** — ALL of the following must be satisfied before declaring Phase 4 complete:
1. Single-step forward+backward passes without error (exit_code=0)
2. Loss **decreases** over 10 steps at smoke config — print step-by-step loss values as evidence
3. Model can overfit 1-2 samples — total loss drops below 1.0 on the fixed batch
4. Two identical runs with the same seed produce **identical** scalar losses
5. No loss term has an anomalous magnitude (>10^6 or NaN) — if any term does, apply the Diagnostic Ladder before proceeding
6. `phase4_report.md` exists with run reports documenting all gate checks above
7. `progress.md` is updated with a `## Phase 4: Integration Proof ✅ COMPLETE` section listing each gate check result

Do not declare Phase 4 complete until ALL checks are documented in `phase4_report.md` and `progress.md` is updated.

---

## Phase 4.5: Integration Audit & Cross-Phase Fix Loop

**Goal**: Verify the implementation matches the paper mathematically, not just structurally. Fix upstream bugs that Phase 4 revealed but could not resolve within `train.py` alone.

**When to enter Phase 4.5**: After Phase 4 exit gate passes but `phase4_report.md` lists anomalies (loss terms that are zero, extremely large, negative where they should be positive, or disabled via lambda=0).

---

### Step 1: Verification Pass — produce `inconsistency_report.md`

Read ALL of the following paper artifacts before writing anything. These are the ground truth sources:

- **PDF**: `paper/CarPlanner.pdf` — primary reference for any ambiguity
- **Equations**: `paper/carplanner_equations.md`
- **Algorithms**: `paper/algorithms.md`
- **Hyperparameters**: `paper/hyperparameters.md`
- **Tables**: `paper/tables.md`
- **Figures**: all `.png` files in `paper/images/` plus their companion `.txt` annotation files

Compare these against the implementation:

1. **Equation verification**: For every equation in `paper/carplanner_equations.md`:
   - Find the code line(s) that implement it
   - Verify the formula matches: correct variables, correct signs, correct operations
   - Write a mapping entry: `Eq N → file.py:line — MATCH / MISMATCH: <description>`

2. **Loss term audit**: For every loss term in the paper's training objective:
   - What does the paper say its typical magnitude should be? (Check tables, figures, or text for any mention of loss scales)
   - What does `phase4_report.md` show?
   - Is the coefficient correct? Is the term actually contributing to `total_loss`?
   - Entry: `loss_term → paper expectation → actual value → MATCH / MISMATCH`

3. **Figure-to-code re-check**: Review `proof.md` mappings. Are there any components in the architecture figures that are present in code but disabled, bypassed, or zeroed out in `implementation/train.py`?

4. **Data scale check**: Print min/max/mean of each input tensor in one batch. Are values at the expected scale for each component? (e.g., if BEV values are 0-255 but the model expects 0-1, that's a scale mismatch)

Write `inconsistency_report.md` with all findings, structured as:

```
# Inconsistency Report

## Equation Mismatches
| Eq # | Paper formula | Code location | Status | Issue |
|------|--------------|---------------|--------|-------|
| Eq 5 | L_value = ... | critic.py:45 | MISMATCH | value target not normalized |

## Loss Term Audit
| Loss term | Paper expectation | Actual value | Coefficient | Status |
|-----------|------------------|--------------|-------------|--------|
| value_loss | ~1.0 (estimated) | 0.0 | 0.5 | BROKEN — not computed |

## Disabled/Bypassed Components
| Component | Why disabled | Paper says |
|-----------|-------------|------------|
| consistency_loss | lambda=0.0 | lambda=0.1 (Table 3) |

## Data Scale Issues
| Tensor | Expected range | Actual range | Status |
|--------|---------------|--------------|--------|
| bev | [0, 1] | [0, 255] | MISMATCH |
```

**Gate**: `inconsistency_report.md` must exist and contain at least the equation and loss term audit sections before proceeding to fixes.

---

### Step 2: Prioritize Fixes

Rank issues from `inconsistency_report.md` by impact:

1. **Critical**: Loss term is zero, NaN, or disabled — the model cannot learn this objective at all
2. **High**: Equation mismatch — the model learns the wrong thing
3. **Medium**: Scale mismatch — the model can learn but converges poorly
4. **Low**: Missing component that the paper includes only in full training (not integration proof)

Work on Critical issues first, then High, then Medium. Skip Low for now.

---

### Step 3: Cross-Phase Fix Loop

For each issue, starting from the highest priority:

#### 3a. Diagnose
- Use the Diagnostic Ladder (see "Run-Report-Diagnose Protocol") to identify which phase and file the bug lives in
- Write a diagnosis in `inconsistency_report.md` under the issue entry:
  ```
  **Root cause**: <phase>/<file.py> — <what is wrong>
  **Fix**: <what needs to change>
  ```

#### 3b. Fix upstream
- You ARE permitted to edit Phase 2 and Phase 3 files during Phase 4.5
- Edit ONLY the file identified in the diagnosis — do not refactor or "improve" adjacent code
- After editing, re-run that file's unit tests:
  ```
  python implementation/test_<submodule>.py
  ```
- If the unit test fails, fix it before returning to integration

#### 3c. Re-run integration
- Run `implementation/train.py --config debug` and compare new loss values to previous
- Update `phase4_report.md` with the new run results
- Update `inconsistency_report.md`: mark the issue as FIXED or STILL BROKEN

#### 3d. Loop limit
- **Max 3 fix attempts per issue**. If the same issue is not resolved after 3 attempts, write a stuck entry:
  ```
  **STUCK**: <issue> — attempted <what>, result was <what>.
  Possible root causes not yet investigated: <list>.
  Recommend: <what a human should check>
  ```
- **Max 3 cross-phase fix iterations total** (not per issue). After 3 full loops through Steps 3a–3c, proceed to Phase 5 regardless, with the stuck report documenting what remains.

---

### Step 4: Exit

**Exit gate** — ONE of the following:

**A. Clean exit** (all issues resolved):
- All loss terms have reasonable magnitudes (no zeros, no >10^6, no negative entropy)
- All Critical and High issues in `inconsistency_report.md` are marked FIXED
- `phase4_report.md` updated with final clean run
- `progress.md` updated with `## Phase 4.5: Integration Audit ✅ COMPLETE`

**B. Stuck exit** (loop limit reached):
- `inconsistency_report.md` contains stuck entries for unresolved issues
- Each stuck entry has a clear diagnosis and recommendation
- `progress.md` updated with `## Phase 4.5: Integration Audit ⚠️ PARTIAL — see inconsistency_report.md`

In either case, proceed to Phase 5.

---

## Phase 5: Benchmarking

**Goal**: Get a working metric and compare to the paper. Prefer ablation parity over headline-number chasing.

#### Phase Brief — write `phase5_brief.md` before running any training

Read the paper's results section, `paper/tables.md`, and `phase1_report.md`. Write `phase5_brief.md` containing:

1. **Target metric**: Exact name, formula, and paper value to reproduce. Which table and row?
2. **Evaluation protocol**: Step-by-step procedure to compute the metric from model outputs — best-of-K sampling count, any filtering, normalisation, GT source, benchmark split name.
3. **Baseline ladder**: List every row in the paper's main results table that you intend to reproduce, in order from simplest to hardest. For each row: configuration name, expected metric value, which components are active.
4. **Ablation table**: If the paper has an ablation, list each ablation row with expected metric delta. This is your primary correctness signal — matching ablation directions is stronger evidence than matching headline numbers.
5. **Figures referenced**: Which figures show training curves, metric vs. epoch, or qualitative results? What trends should you expect to see?
6. **Hardware adaptation**: If paper used different hardware, note any changes to batch size, learning rate scaling, or step count, and the paper section that justifies each.

**Forbidden entries in `phase5_brief.md`**: "metric TBD", "evaluation steps to be confirmed", "ablation table not available" — query the PDF and extract the actual values before writing the brief.

**Brief gate**: `phase5_brief.md` must exist and contain the target metric value (exact number from paper), the exact evaluation protocol steps (quoted from paper), and the ablation table before starting any training run.

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
Run full config twice. Results must be within tolerance (define tolerance in `phase1_report.md`). If not, the result is exploratory — not evidence.

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
4. **High-impact ambiguities**: From `phase1_report.md`'s ambiguity register, which medium/high-impact items have not yet been tested? List the alternative interpretations to try.
5. **Stop criteria**: Exact numeric targets that define "close enough" — derived from `phase1_report.md` tolerance definition.

**Forbidden entries in `phase6_brief.md`**: "gap TBD", "hypothesis to be determined", "improvement unclear" — base every hypothesis on a specific paper section, equation, or figure.

**Brief gate**: `phase6_brief.md` must exist and reference the current metric gap with specific numbers and at least two paper-grounded improvement hypotheses (each citing a paper equation or figure) before starting the experiment loop.

---

Each experiment:

1. Compare metric to paper baseline (check direction from `requirements.md`).
2. Diagnose the gap. Use logged diagnostics, not guesswork. Check:
   - Do loss components match expected magnitudes from the paper?
   - Are intermediate representations healthy (reasonable mean/std, diverse across inputs)?
   - Does the evaluation protocol exactly match `phase1_report.md`?
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
- Metric within tolerance of paper baseline (tolerance defined in `phase1_report.md`).
- Remaining gap is isolated and documented — you know exactly which component/ambiguity is responsible but cannot resolve it without human input.


**Stuck after 5+ failed experiments:**
- Re-read the paper with your code open. Map each equation to its code line.
- Re-read `failure_patterns.md`.
- Revisit high-impact ambiguities in `phase1_report.md` — try the alternative interpretation.
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

---

## Run-Report-Diagnose Protocol

This protocol applies to every phase that runs code (Phases 2–6). It replaces blind edit-retry loops with structured observation and diagnosis.

### 1. Post-Run Report

After every significant run (integration test, smoke test, full training), write or append to `phaseN/phaseN_report.md`:

```
## Run: <config> — <date or turn>
**Command**: <exact command run>
**Exit code**: <0 or 1>
**Loss values** (every term, not just total):
- policy_loss: <value>
- value_loss: <value>
- entropy: <value>
- consistency_loss: <value>
- total_loss: <value>
**Gate checks**:
- [ ] Loss decreases over N steps
- [ ] Overfit test passes
- [ ] Reproducibility matches
- [ ] <any phase-specific gate>
**Anomalies**: <anything unexpected — extreme values, NaN, wrong signs, unexpected magnitudes>
```

This report survives history pruning and prevents re-running identical experiments. Always write the report before attempting any fix.

### 2. Diagnostic Brief (before any code edit)

When a run fails or a gate check does not pass, **do NOT edit any code yet**. First, append a diagnostic section to `phaseN/phaseN_report.md`:

```
## Diagnosis: <one-line symptom>
**Symptom**: <what went wrong — exact values, exact error message>
**Hypotheses** (ranked by likelihood):
1. <most likely cause> — evidence: <why you think this>
2. <next most likely> — evidence: <why>
3. <least likely> — evidence: <why>
**Investigation plan**:
- [ ] Read <file.py> lines N–M to check <specific thing>
- [ ] Run <command> to verify <hypothesis>
**Suspected phase/layer**: <Phase N — which phase's code likely contains the bug>
```

Only after writing this diagnosis may you begin investigating and editing code.

### 3. Diagnostic Ladder

When investigating a failure, work **backward through the pipeline** from the symptom. Do NOT rewrite the current phase's file repeatedly — the bug may live in an earlier phase.

**Investigation order** (check each layer before moving to the next):

1. **Current phase wiring** — is the integration code (e.g. `train.py`) passing the right tensors, in the right order, with the right dtypes?
2. **Loss assembly** — do coefficients match the paper? Is the formula correct? Print each term's magnitude on step 1.
3. **Individual submodules** (Phase 3) — run each submodule's test file in isolation (`python implementation/test_<submodule>.py`). Does it still pass? Are output magnitudes reasonable?
4. **Data pipeline** (Phase 2) — are input values at expected scale? Are targets normalised? Print min/max/mean of one batch.
5. **Paper contract** (Phase 1) — re-read the relevant equation in the paper and verify the contract matches.
6. **Paper extraction** (Phase 0) — was the equation or figure extracted correctly?

**Rules**:
- Do not edit code until you have identified which layer the bug is in.
- Do not edit more than one layer per fix attempt.
- After each fix, re-run and update the report. If the symptom changes, write a new diagnosis.
- If 3 fix attempts at the same layer fail, move one layer deeper in the ladder.

### 4. Common Symptoms → Where to Start

| Symptom | Start at |
|---|---|
| Loss is NaN | Layer 2: loss assembly (log of zero? division by zero? missing gradient clipping?) |
| Loss extremely large (>10^6) | Layer 3: submodule output scale → Layer 4: data value scale → Layer 2: loss coefficients |
| Loss doesn't decrease | Layer 2: learning rate / optimizer → Layer 3: frozen params that should be trainable → Layer 2: loss sign |
| Loss decreases but metric bad | Evaluation protocol → train/eval mismatch → metric computation |
| Shape mismatch error | Layer 1: integration wiring → Layer 3: submodule contract |
| Overfit test fails | Layer 3: model capacity → Layer 4: data loader returning different samples each time → label correctness |
| Reproducibility fails | Non-deterministic ops → unseeded randomness → data loader shuffle seed |



to the instruction set for the paper implementor,we added these specific tests: 1. Figure extractor from the paper. all the figures should be clipped and stored in a
seperate folder called images. 2. Each figure should be accompanied by a text file which includes the caption for the figure, extract textual information in the    
figure, and extracts pertaining text from the paper pertaining the figure- say architectural details, dimensions etc. 3. all equation listed should eb extracted    
from the figure into a file called equations. 4. referenced test metrics should be extracted too.