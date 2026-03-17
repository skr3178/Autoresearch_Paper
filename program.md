# autoresearch-paper

Autonomous agent framework for implementing research papers from scratch.

---

## Before You Start

Read these files completely before writing any code:

1. `requirements.md` — paper summary, dataset, target metric, reported baseline
2. `failure_patterns.md` — general-purpose failure catalog from past implementations
3. `paper/` — the paper PDF and any supplementary material

You build everything from scratch inside `implementation/`. There is no existing code to modify.

---

## Setup

1. **Run tag**: propose a tag based on today's date (e.g. `paper-mar17`). Branch `autoresearch/<tag>` must not exist.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read everything**: `requirements.md`, `failure_patterns.md`, paper in `paper/`.
4. **Decompose the paper** (see "How to Decompose" below).
5. **Write `progress.md`**: ordered component checklist with success criteria per component.
6. **Initialize `results.tsv`** with the header row.
7. **Commit**: `git add -A && git commit -m "setup: implementation plan"`

---

## How to Decompose a Paper

**Do NOT decompose by paper sections.** Papers are structured for readability, not implementability. Instead:

### Step 1: Identify the core learning component
Every paper has one central mechanism that makes it work. Find it. This is what you build and verify first — everything else is downstream.

### Step 2: Order by dependency and risk
Build dependencies before dependents. Build risky/uncertain components before safe ones. If the core mechanism doesn't work, nothing else matters — so validate it early.

### Step 3: Start with the simplest ablation
If the paper has an ablation table, implement the simplest variant first (e.g. single modality, no auxiliary losses, smallest model). This gives you a working baseline before you add complexity. The ablation table tells you which components contribute most — add those first.

### Step 4: Plan for staged training
If the paper has multiple components trained jointly, plan to train them separately first. Joint training often fails due to gradient conflicts. A common pattern: train component A → freeze A → train component B on A's outputs. Only attempt joint training after both work independently.

### Step 5: Multiple config tiers
Define configs for different scales. You will use these throughout development:
- **debug**: smallest possible dimensions (2 layers, 64 hidden, batch 2, 10 steps). Runs in <5 seconds. Used to catch shape/dtype/import errors.
- **smoke**: small but realistic (fewer layers, small batch). Runs in 1-2 minutes. Used to verify loss decreases.
- **full**: paper-scale or hardware-adapted. Used for real training runs.

---

## Implementation Loop

Implement one component at a time, in the order defined in `progress.md`.

### Step 1: Understand before coding
- Re-read the paper section for this component.
- Create a mapping table: paper notation → code variable name → meaning. Papers use math notation that doesn't map obviously to code. Making this explicit prevents coefficient inversions, wrong signs, and misattributed terms.
- If the paper has a multi-step algorithm, write out every step as a numbered comment before coding any of them.
- Check `failure_patterns.md` for relevant warnings.

### Step 2: Implement minimally
- Create `implementation/<component>.py`.
- Parameterize all dimensions from a config dict — never hardcode spatial sizes, channel counts, sequence lengths, or vocabulary sizes. You will need to run at different scales.
- When the paper is ambiguous (missing hyperparameter, unclear architecture detail): implement the simplest interpretation, make it configurable, and add a comment noting the ambiguity and your choice.

### Step 3: Test before moving on
Write `implementation/test_<component>.py` with:

1. **Shape assertions**: verify every tensor shape matches what you expect from the paper.
2. **Dtype assertions**: verify no silent type promotions (e.g. int16 → int64 through masking, float16 → float32 through custom ops).
3. **Forward + backward pass** at debug config: verifies the component can train.
4. **Output diversity check**: feed two different inputs → outputs should differ meaningfully. If outputs are near-identical (cosine similarity >0.95), something is collapsing.
5. **Loss finiteness**: if this component produces a loss, verify it's finite and in a reasonable range.

Run: `uv run implementation/test_<component>.py`

### Step 4: Decide
- **PASS**: Mark `✅` in `progress.md`, `git commit`.
- **FAIL**: Debug (max 3 attempts). If stuck, mark `⚠️ BLOCKED` with notes and move on.

---

## Integration

Once components are individually tested:

1. Wire everything into `implementation/train.py`.
2. **Single-step smoke test** at debug config: one forward + backward pass. This catches cross-module shape/dtype mismatches and import errors.
3. **10-step smoke test** at smoke config: verify loss decreases (even slightly). If loss is flat or increasing after 10 steps, something is wrong — do not proceed to full training.
4. **Profile the data loader**: load 10 batches, time it. If >2 seconds for 10 batches at batch_size=2, the loader is too slow — vectorize any per-element loops before proceeding.
5. `git commit` working integration.

---

## Training

### First run
Run training at smoke config first, not full scale:
```
uv run implementation/train.py --config smoke > run.log 2>&1
```

Verify:
- Loss decreases over the run
- No NaN or Inf values
- Memory usage is reasonable
- Outputs at end of training look plausible (if applicable)

Only after smoke config works, run at full config with the time budget from `requirements.md`:
```
uv run implementation/train.py --config full > run.log 2>&1
```

### What to log during training
Log these every N steps (N ≈ 100):
- **Each loss component separately** — not just total loss. When total loss misbehaves, you need to know which term is responsible.
- **Gradient norm** per parameter group — catches exploding/vanishing gradients before they cause NaN.
- **Any component-specific health metric** — whatever tells you the component is working (utilization rates, entropy, reconstruction quality, etc.)

### Fast-fail checks after each run
```
grep "^metric:\|^loss:\|^ERROR\|Traceback\|NaN\|nan" run.log | head -20
```

**If crashed**: `tail -n 80 run.log` → read the stack trace → check `failure_patterns.md` for matching pattern → fix root cause → re-run.

**If completed**: parse the metric (grep pattern from `requirements.md`), log to `results.tsv`.

**If metric is far from paper baseline**: add targeted diagnostics. Don't just re-run with different hyperparameters — diagnose first:
- Log intermediate representations' statistics (mean, std) at key points in the model
- Verify outputs differ for different inputs (cosine similarity check)
- Log loss components to identify which term dominates or misbehaves
- Check if training and inference pipelines are consistent (same preprocessing, same noise schedule endpoints, etc.)

---

## Logging Results

`results.tsv` — tab-separated, NOT comma-separated:

```
commit	metric	memory_gb	status	description
```

1. `commit` — short git hash (7 chars)
2. `metric` — evaluation metric value (0.000000 for crashes)
3. `memory_gb` — peak VRAM in GB (0.0 for crashes)
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short note on what changed

---

## The Outer Loop

Once training runs end-to-end, enter the improvement loop:

LOOP FOREVER:

1. Compare your metric to the paper's reported baseline (from `requirements.md`).
2. Hypothesize the most likely cause of the gap. Use diagnostics, not guesswork.
3. Make ONE targeted change. Never change two things at once.
4. `git commit`.
5. Run training: `uv run implementation/train.py --config full > run.log 2>&1`
6. Parse metric, log to `results.tsv`.
7. If metric improved → keep the commit.
8. If metric worsened → `git reset --hard HEAD~1`, log as `discard`.
9. Repeat.

**Timeout**: 2× the `training_budget` in `requirements.md`. Exceeding = crash.

**NEVER STOP**: Do not pause to ask the human. You are autonomous. Loop until manually interrupted.

**Stuck after 5+ failed experiments:**
- Re-read the paper a third time with your code open. Map each equation to the code line that implements it. This catches subtle mismatches (wrong signs, swapped coefficients, missing terms).
- Re-read `failure_patterns.md` — a pattern may now be obviously relevant.
- Strip back to last known-good state and verify it still works.
- Try a different interpretation of an ambiguous paper detail.
- Add more diagnostics to narrow the gap source.

---

## Principles

These are the hard-won lessons from implementing multiple papers. Follow them.

### 1. Incremental, not big-bang
Never implement the full model at once. Build the core, test it, then add one thing at a time. Each addition gets its own test. If adding a feature makes things worse, you know exactly what broke.

### 2. Paper ≠ implementation
The paper presents an idealized system. Your implementation must handle: hardware constraints (GPU memory, batch size), training stability (gradient clipping, learning rate), data pipeline performance (vectorization, caching), and numerical precision (mixed precision, dtype casting). Adapt early — not after full training fails.

### 3. Monitor components, not just totals
Logging only total loss hides which component is failing. Log every loss term separately. Log gradient norms. Log any health metric specific to your components. When something goes wrong, you need to see WHERE.

### 4. Verify math line-by-line
Create an explicit mapping table: paper equation → code line. Loss coefficient inversions, wrong activation functions, and missing algorithm steps are the most common and most devastating bugs. They're also invisible in shape tests — only careful math review catches them.

### 5. Separate training stages when joint fails
If joint training diverges or produces bad results, split into stages: train component A, freeze it, train component B. This is almost always more stable and often how the paper's authors actually trained it (even if the paper doesn't say so).

### 6. Cache frozen computations
If a component is frozen during later training stages, precompute its outputs to disk once. Never re-run a frozen component inside the training loop. This often provides 10-100× speedup.

### 7. Test with simple data first
If possible, validate your pipeline on simple/synthetic data before real data. Simple data lets you visualize outputs, verify correctness by eye, and iterate in seconds instead of hours.

### 8. Make everything configurable
When the paper is ambiguous, don't hardcode your guess. Make it a config parameter with a comment explaining the ambiguity. You'll need to try different values.

### 9. Read the ablation table
The ablation table tells you what matters and what doesn't. Implement high-impact components first. If a component provides <1% improvement in the paper, defer it.

### 10. Profile before optimizing
Measure where time is actually spent before optimizing. Often the bottleneck is data loading (Python loops over spatial data), not model computation. A 50× speedup in the data loader matters more than any model optimization.
