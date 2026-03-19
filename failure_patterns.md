# Failure Patterns

General-purpose failure catalog distilled from implementing multiple research papers. Organized by failure category, not by specific bug. Read this in full before starting any implementation.

---

## Category 1: Math-to-Code Translation Errors

The most devastating class of bugs. They pass all shape tests, produce plausible-looking loss values, and silently destroy your results. The only defense is line-by-line verification of equations against code.

### 1.1 Loss Coefficient Inversion
**What happens**: Two coefficients in a loss function get swapped. The code runs, loss decreases, but the model learns the wrong thing.
**Why it happens**: Papers write `λ₁ · L_a + λ₂ · L_b` but don't always make it obvious which λ corresponds to which semantic role. Code uses variable names like `cost_a` and `cost_b` that may not match the paper's subscripts.
**Prevention**: Create an explicit mapping table for every loss function:
```
Paper symbol → Code variable → Semantic meaning → Value
λ₁           → alpha_recon   → reconstruction     → 1.0
λ₂           → beta_reg      → regularization      → 0.25
```
Print each term's magnitude on the first training step. Verify the ratio makes sense.

### 1.2 Wrong Activation Before Loss
**What happens**: Adding an activation function (sigmoid, softmax) before a loss that already applies it internally. Result: double activation, collapsed output range, numerically unstable training.
**Why it happens**: The model's output layer has an activation, and the loss function also applies one. Common pairs:
- `nn.Sigmoid()` + `BCEWithLogitsLoss` (applies sigmoid internally)
- `nn.Softmax()` + `CrossEntropyLoss` (applies log-softmax internally)
**Prevention**: Before writing any loss computation, check the PyTorch docs for the loss function's expected input format. If it says "logits", the model should NOT apply the activation.

### 1.3 Algorithm Step Omission
**What happens**: A multi-step algorithm from the paper is implemented with a step missing or collapsed into another step. Particularly common in iterative/sampling algorithms.
**Why it happens**: Steps that seem "obvious" or "trivial" get skipped. But in iterative algorithms, every step exists for a reason.
**Prevention**: Write every step from the paper as a separate numbered comment in code, citing the algorithm number and step. Do not collapse steps. Test on a tiny example where you can trace intermediate values by hand.

### 1.4 Wrong Sign or Direction
**What happens**: A gradient, residual connection, or update step has the wrong sign. Loss decreases but model learns the opposite of what's intended.
**Prevention**: For any additive update: `x = x + delta` vs `x = x - delta` — verify against the paper. For masking: causal mask (upper triangle) vs anti-causal — verify which objective requires which direction.

---

## Category 2: Implementing Too Much at Once

The single most consistent failure pattern across all implementations. LLMs try to implement the full paper in one pass, leading to complex bugs that are hard to isolate.

### 2.1 Big-Bang Integration
**What happens**: All components are built and wired together before any individual testing. When training fails, it's impossible to tell which component is broken.
**Prevention**: Build and test ONE component at a time. Each gets its own test file. Only integrate after individual tests pass.

### 2.2 Skipping the Simplest Ablation
**What happens**: Full multi-modal / multi-loss model is built first instead of the simplest working variant.
**Prevention**: Check the paper's ablation table. Implement the simplest row first (fewest components, single modality, no auxiliary losses). This is your baseline. Add components one at a time and verify each addition improves the metric.

### 2.3 Joint Training Before Staged Training
**What happens**: Components that can be trained separately are trained jointly from the start. Joint training has more failure modes (gradient conflicts, learning rate sensitivity, collapse of one component).
**Prevention**: Default to staged training: train A → freeze A → train B using A's outputs. Attempt joint training only after both work independently.

---

## Category 3: Hardware and Performance Blindness

### 3.1 Using Paper Hyperparameters Directly
**What happens**: Paper says batch_size=16, you set batch_size=16, training OOMs on your GPU.
**Why it happens**: Papers are typically run on A100/H100 clusters. Your GPU has less memory.
**Prevention**: Create config tiers (debug/smoke/full). Start at debug. Scale up only after verifying the code works. Adjust batch size, model size, and gradient accumulation for your hardware.

### 3.2 Unvectorized Data Processing
**What happens**: Data loading/preprocessing uses Python for-loops over spatial positions (pixels, points, grid cells). Each sample takes hundreds of milliseconds. Training is bottlenecked on data, not compute.
**Prevention**: Profile your data loader before starting full training. If 10 batches at batch_size=2 take >2 seconds, the loader needs vectorization. Replace for-loops with numpy broadcasting, `torch.scatter`, `np.where`, or similar.

### 3.3 Recomputing Frozen Components
**What happens**: A component frozen during stage-2 training is still called inside the training loop, recomputing the same outputs every step.
**Prevention**: If a component is frozen, precompute ALL its outputs to disk before the training loop. Load from disk in the dataset. Expected speedup: 10-100×.

### 3.4 Mixed Precision Traps
**What happens**: Custom operations that don't support float16 produce NaN when `torch.autocast` is active. Or: values cached in float16 lose precision when loaded later.
**Prevention**: Run your first training attempt WITHOUT mixed precision. Add it later as an optimization. When you do add it: use gradient clipping, and verify that any cached/precomputed tensors are stored in float32.

---

## Category 4: Silent Data Bugs

### 4.1 Dataset Indexing Type Mismatch
**What happens**: Code assumes dict-style access (`batch['image']`) but dataset returns tuples (or vice versa). Crashes on first batch.
**Prevention**: Before writing any training loop, print the type and structure of one sample:
```python
sample = dataset[0]
print(type(sample), sample.keys() if hasattr(sample, 'keys') else len(sample))
```

### 4.2 Coordinate System Confusion
**What happens**: Data uses one coordinate convention (e.g. Z-up) but code assumes another (e.g. Y-up). Or: GPS coordinates used where local coordinates are expected.
**Prevention**: Load one sample, visualize it, verify the axes are what you expect. For any coordinate transform: test with a known point and verify the output.

### 4.3 Data Split Mismatch
**What happens**: Paper uses specific train/val/test splits but implementation uses different splits, making metric comparison meaningless.
**Prevention**: Cross-reference the paper's experimental section against your data loading code. Verify split sizes match.

### 4.4 Tokenizer/Vocabulary Mismatch
**What happens**: Model's vocabulary size doesn't match the tokenizer's vocabulary size. Outputs are systematically wrong.
**Prevention**: Assert at model initialization:
```python
assert model.vocab_size == tokenizer.vocab_size
```
For factorized tokenizers: verify the number of factors, not just total vocabulary.

---

## Category 5: Training Dynamics Failures

### 5.1 Training-Inference Mismatch
**What happens**: Training sees one data distribution but inference starts from a different one. Common with noise schedules: training adds mild noise, inference starts from pure noise.
**Prevention**: Verify that the endpoints match. For diffusion: the final noise level in the forward process should match the starting point of the reverse process. For masking: the mask ratio range during training should cover the fully-masked case used at inference.

### 5.2 Component Collapse
**What happens**: One component of the model stops contributing — its outputs become constant or near-identical across different inputs.
**Symptoms**: Near-identical outputs for different inputs (cosine similarity >0.95). Utilization metrics drop (if applicable). One loss term stops decreasing while others continue.
**Prevention**: Periodically check output diversity: feed multiple different inputs, compute pairwise cosine similarity. If >0.95, the component is collapsing. Investigate the specific component, not just the overall model.

### 5.3 Logging Only Total Loss
**What happens**: Only total loss is logged. When training fails or stalls, it's impossible to tell which loss component is responsible.
**Prevention**: Log every loss term separately. Always. If loss has 3 terms, log 3 values plus the total. When debugging, plot each term independently.

### 5.4 Gradient Issues Hidden Until Scale
**What happens**: Gradients work fine at debug scale but explode or vanish at full scale (more layers, longer sequences, larger batch).
**Prevention**: Log gradient norms per parameter group during the first N steps at each scale tier. Add gradient clipping from the start (it's cheap and prevents catastrophic failures).

---

## Category 6: Ambiguity Mishandling

### 6.1 Hardcoding an Interpretation
**What happens**: Paper is ambiguous about a detail. The implementation hardcodes one interpretation. Later, when results don't match, it's unclear whether the interpretation is wrong or something else is broken.
**Prevention**: Make ambiguous details configurable. Add a comment citing the paper section and noting the ambiguity. Try the simplest interpretation first.

### 6.2 Not Checking What the Data Actually Contains
**What happens**: Paper says "we use dataset X" but doesn't specify exact format, split sizes, or preprocessing. Implementation assumes a format that doesn't match reality.
**Prevention**: Before implementing any data loading: load one sample from disk, inspect its type, shape, dtype, and value range. Plot/visualize it if possible. Only then write the data pipeline.

### 6.3 Missing Dimension Parameterization
**What happens**: Spatial dimensions, sequence lengths, or channel counts are hardcoded for the paper's setting. When running at a different scale (debug config), the code breaks.
**Prevention**: Every dimension that could change should come from a config object. Use the config in all shape computations, reshapes, and linear layer definitions.

---

## Category 7: Checkpoint and Compatibility Failures

### 7.1 Checkpoint Key Inconsistency
**What happens**: `KeyError` or missing keys when loading a checkpoint. Training resumes from scratch or crashes.
**Why it happens**: Different codebases, training stages, or save utilities use different key conventions — `"model"` vs `"model_state_dict"` vs `"state_dict"`, or nested vs flat dicts.
**Prevention**: Always inspect checkpoint keys before loading:
```python
ckpt = torch.load(path, map_location="cpu")
print(list(ckpt.keys()))
```
Write a `load_checkpoint()` helper that handles the key conventions you encounter, rather than hardcoding one.

### 7.2 Architecture Change Invalidates Old Checkpoints
**What happens**: After fixing a bug in the model (e.g. removing an incorrect activation, changing a layer), old checkpoints produce degraded outputs — collapsed representations, wrong outputs, or worse metrics than before the fix.
**Why it happens**: Bug fixes change weight semantics. Weights trained with the bug learned to compensate for it. After the fix, those compensations become harmful.
**Prevention**: After any architecture fix that changes the forward pass behavior, retrain from scratch. Document in `progress.md` which commits are checkpoint-compatible. Never assume old weights are valid after changing the model's computation graph.

---

## Category 8: Evaluation and Benchmark Mismatch

Metric looks wrong not because the model is wrong, but because the evaluation doesn't match the paper's protocol. These bugs don't crash — they produce numbers that are subtly incomparable to the paper.

### 8.1 Best-of-K vs Single Prediction
**What happens**: Paper reports best-of-K metric (generate K predictions, report the best) but implementation evaluates a single prediction. Result is systematically worse than paper for the wrong reason.
**Prevention**: Check the paper's evaluation section for "best-of-K", "min over K samples", "oracle" language. Document in evaluation contract whether the metric is single-sample or best-of-K, and what K is.

### 8.2 Paper Split vs Local Split
**What happens**: Paper's test set has N samples, your test set has M ≠ N. Metric is computed over different data, making comparison meaningless.
**Prevention**: Verify split sizes match the paper's experimental section. If using a subset, document the expected metric adjustment in `paper_contract.md`.

### 8.3 Raw GT vs Reconstructed GT
**What happens**: Paper computes metric against raw ground truth, but implementation computes against ground truth that has been through an encode-decode cycle (e.g. tokenized then detokenized). The reconstruction error is baked into the "ground truth", inflating the metric.
**Prevention**: Verify the source of ground truth used in metric computation. If any preprocessing is applied to GT, it must match what the paper does.

### 8.4 Unit or Scale Mismatch
**What happens**: Metric is computed in different units (meters vs centimeters, radians vs degrees, pixels vs normalized coordinates). Off by a constant factor.
**Prevention**: Check units of both predictions and ground truth before metric computation. Print a few values from each and verify they're in the same range.

### 8.5 Sampling/Inference Schedule Mismatch
**What happens**: Paper uses N diffusion steps / sampling iterations at inference, implementation uses a different N. Or noise schedule endpoints differ between training and inference.
**Prevention**: Cross-check inference procedure against training. Document in evaluation contract: number of sampling steps, schedule type, and endpoints.

### 8.6 Threshold or Filtering Mismatch
**What happens**: Paper applies distance thresholds, confidence filters, or ROI restrictions during evaluation that the implementation misses (or applies differently).
**Prevention**: Look for phrases like "within X meters", "with confidence > Y", "in the ROI" in the paper's evaluation section. Document every filter in the evaluation contract.

### 8.7 Metric Name Reused with Different Formula
**What happens**: Paper says "FID" or "ADE" but uses a variant formula (different feature extractor, different normalization, different averaging). Implementation uses the standard formula, gets different numbers.
**Prevention**: Check if the paper references a specific implementation or library for metric computation. If they cite someone else's metric code, use the same code.

---

## Category 9: False Proof

Tests pass, loss decreases, outputs look plausible — but the implementation is still wrong. These are the hardest bugs to catch because they bypass normal verification.

### 9.1 Shapes Pass, Equation is Wrong
**What happens**: All shape assertions pass, forward/backward works, but the mathematical operation is incorrect (wrong coefficients, wrong sign, missing term). The model trains but learns the wrong thing.
**Why shapes don't catch it**: Shape tests verify tensor dimensions, not tensor contents. A matrix multiply with the wrong weight matrix has the same output shape as the correct one.
**Prevention**: Equation oracle tests — hand-compute the expected output for a tiny input and assert the code matches. This is the only reliable defense.

### 9.2 Short Overfit Passes, Inference is Wrong
**What happens**: Model overfits training data, loss goes to near-zero, but generated outputs at inference are wrong (blurry, repetitive, collapsed).
**Why it happens**: Training and inference use different code paths (different noise levels, different sampling, different masking). The training path works but the inference path has a bug.
**Prevention**: After overfitting on 1-2 samples, run the full inference pipeline on those same samples. The generated output should closely match the training data.

### 9.3 Metric Improves for the Wrong Reason
**What happens**: Metric gets better after a change, but the improvement comes from a bug that happens to help the metric (e.g. test-time data leakage, evaluation on training data, accidentally easier test set).
**Prevention**: When a change produces surprisingly large improvement, be suspicious. Verify the evaluation is on the correct split. Check that the improvement is consistent across multiple seeds.

### 9.4 Visualization Looks Plausible but is Wrong
**What happens**: Generated images/trajectories/videos look reasonable to a human eye but have subtle errors (wrong coordinate frame, flipped axes, systematic offset).
**Prevention**: Compare against ground truth quantitatively, not just visually. Overlay predictions on ground truth. Compute point-wise error and visualize the error map.

### 9.5 Pretrained Checkpoint Masks Implementation Bugs
**What happens**: Loading pretrained weights produces good results, so the code appears correct. But the code has a bug that the pretrained weights happen to compensate for. When training from scratch, the bug surfaces.
**Prevention**: Always validate the pipeline with training from scratch, not just with pretrained weights. A pretrained checkpoint is not evidence of code correctness.

---

## Category 10: Autonomous Loop Pathologies

Failure modes specific to autonomous agents running without human oversight.

### 10.1 Endless Hyperparameter Search Without Diagnosis
**What happens**: The agent runs dozens of experiments tweaking learning rate, batch size, weight decay — without ever diagnosing WHY the metric is wrong. The real problem is a code bug, not hyperparameters.
**Prevention**: Before changing any hyperparameter, diagnose first. Log loss components, gradient norms, intermediate representations. If the model can't overfit 1-2 samples, no hyperparameter change will help.

### 10.2 Deleting Failed Experiment History
**What happens**: Agent uses `git reset --hard` to remove failed experiments. History is lost. The same failed experiments may be repeated. Debugging information is destroyed.
**Prevention**: Never delete commits. Use `git revert` if needed. Log every experiment in `results.tsv` and `decision_log.md`, including failures.

### 10.3 Changing Multiple Variables Per Experiment
**What happens**: Agent makes 3 changes at once. Metric improves. It's impossible to know which change helped (or whether one change helped and two hurt, with a net positive).
**Prevention**: One change per experiment. Always. If you want to try a combination, first test each change individually.

### 10.4 Moving On Without Isolating Uncertainty
**What happens**: A component partially works (loss decreases but metric is off). Agent moves to the next component instead of fully diagnosing the issue. Later, the unresolved issue compounds with new bugs and becomes impossible to isolate.
**Prevention**: Before moving on from a component, document the remaining uncertainty in `proof.md`. If the uncertainty is high-impact, resolve it before building on top of it.

### 10.5 Treating "Training Completed" as Evidence of Correctness
**What happens**: Training finishes without crashing, loss decreases, agent concludes the implementation is correct. But the model learned the wrong thing (wrong loss coefficients, wrong evaluation protocol, wrong data preprocessing).
**Prevention**: "It runs" is not proof. "Loss decreases" is not proof. The bar is: ablation deltas match the paper, evaluation protocol matches the contract, and at least one metric value is in the plausible range.

---

## Category 7: Debugging Anti-Patterns

### 7.1 Weakening Tests Instead of Fixing the Implementation
**What happens**: A test fails. Instead of tracing the failure to its root cause in the implementation, the test assertion is changed or removed to make the test pass. The bug remains in the code.
**Why it happens**: It's faster to fix the test than to understand the code. The agent sees the assertion line as the "thing that's failing" rather than evidence of a bug.
**Prevention**: When a test fails, the test is correct until proven otherwise. The failure message is a symptom — trace it back to the implementation. Fix the implementation, not the test. Only change a test if you can prove the test itself was wrong (e.g. wrong expected shape from the paper contract).

### 7.2 Rewriting Instead of Patching
**What happens**: When a bug is found, the entire file is rewritten rather than patching the specific line. This introduces new bugs and loses previous correct logic.
**Prevention**: When a test fails, read the traceback, find the exact line in the implementation that produced the wrong value, and change only that line. If rewriting more than 5 lines to fix a single test failure, stop — you are likely solving the wrong problem.

### 7.3 Wrong Python Environment
**What happens**: Using the wrong venv causes `ModuleNotFoundError` for either `torch` or `nuplan`. There are THREE venvs — each for a different purpose.
**Prevention**: Two venvs exist — use the right one:
- **Torch / CUDA / model training**: `/media/skr/storage/autoresearch/.venv/bin/python <script>`
- **nuPlan data loading (imports nuplan.*)**: `/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python <script>`

Never use `uv run python` — it has neither torch nor nuplan. Never attempt `pip install` for either — both are already installed in their respective venvs.

### 7.4 Dimension Confusion in Reshape
**What happens**: `reshape(-1, C, H, W)` is used when the input is a flat array for a single sample. The `-1` infers a size-1 leading dimension, producing shape `(1, C, H, W)` instead of `(C, H, W)`. Downstream, the batch dimension collides with this extra dim.
**Prevention**: For a single sample from a byte buffer, use explicit shapes: `reshape(C, H, W)`. Only use `-1` when you know a batch dimension exists in the buffer.

### 7.7 Mixing nuPlan and Torch in One Script
**What happens**: The data loader imports both `nuplan.*` and `torch`. No single venv has both — the nuplan venv has no torch, the torch venv has no nuplan. The agent strips out one to make the other work, producing a broken data loader.
**Prevention**: Split data loading into two stages:
1. **Extraction script** `implementation/extract_nuplan.py` — runs with nuplan venv, reads `.db` files via nuplan API, saves scenarios as numpy `.npz` files to `implementation/cache/`
2. **PyTorch Dataset** `implementation/data_loader.py` — runs with torch venv, reads `.npz` files, converts to tensors. No nuplan imports.
Run extraction once: `/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit/nuplan_venv/bin/python implementation/extract_nuplan.py`
Then run training with: `/media/skr/storage/autoresearch/.venv/bin/python implementation/train.py`

### 7.5 Querying nuPlan SQLite Directly
**What happens**: The agent writes a raw `sqlite3` connection querying made-up tables like `scenarios` with columns `bev`, `ego_history`, `gt_trajectory`. These tables do not exist. nuPlan's schema uses `lidar_pc`, `ego_pose`, `scene`, `track`, etc. — and even those should not be queried directly.
**Prevention**: Always use the nuPlan devkit API (`NuPlanScenario`, scenario builders) to load data. The devkit handles the schema. Never use `sqlite3` directly on nuPlan `.db` files. See `requirements.md` for the devkit path and usage.

### 7.6 Hardcoding Placeholder Paths
**What happens**: The agent writes a placeholder path like `/path/to/nuplan.db` in code instead of reading `requirements.md` for the actual dataset location. Tests then fail with `FileNotFoundError` or `sqlite3.OperationalError`.
**Prevention**: Always read `requirements.md` before writing any file I/O code. The dataset paths are specified there. Use exactly those paths — never invent placeholders.

---

## Pre-Training Checklist

Before starting any training run, verify:

- [ ] Mapping table created for every loss function (paper symbol → code variable → value)
- [ ] No activation function before a loss that applies it internally
- [ ] Every multi-step algorithm has numbered comments matching the paper
- [ ] Debug config forward + backward pass works (<5 seconds)
- [ ] Smoke config loss decreases over 10 steps
- [ ] Data loader profiled: <2 seconds for 10 batches at batch_size=2
- [ ] All frozen components precomputed to disk (if applicable)
- [ ] Each loss component logged separately
- [ ] Gradient norms logged per parameter group
- [ ] Output diversity verified (different inputs → different outputs)
- [ ] Dataset split matches paper's experimental section
- [ ] All dimensions parameterized from config (no hardcoded sizes)
