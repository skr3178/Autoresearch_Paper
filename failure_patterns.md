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
