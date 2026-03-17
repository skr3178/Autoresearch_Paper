# Paper Requirements

Fill in this file before handing off to the agent. The agent reads this as the primary specification.

---

## Paper

**Title:** <!-- e.g. "CoPilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion" -->

**Summary:** <!--
2–5 sentences describing what the paper proposes:
- What problem does it solve?
- What is the key architectural idea?
- What makes it novel compared to prior work?
-->

**Key paper sections to implement:**
<!-- List the sections/algorithms the agent must implement. Be explicit.
Example:
- Section 3.1: VQ-VAE tokenizer (Algorithm 1)
- Section 3.2: Discrete diffusion model (Algorithm 2, 3)
- Section 4: Training procedure
-->

---

## Dataset

**Name:** <!-- e.g. "TinyStories", "Moving MNIST", "nuScenes BEV" -->

**Location:** <!-- Absolute path or HuggingFace dataset ID
Example: /media/skr/storage/datasets/nuscenes
Example: roneneldan/TinyStories
-->

**Format:** <!-- What does one sample look like?
Example: "Each sample is a dict with keys 'image' (3×256×256 float32) and 'label' (int)"
Example: "Each sample is a 17-frame video clip at 256×256, stored as uint8 PNG files"
-->

**Preprocessing required:** <!-- Any steps the agent must implement to prepare data
Example: "Tokenize with pre-trained VQ-VAE at resolution 64×64"
Example: "Normalize to [-1, 1], center-crop to 256×256"
-->

**Train / val split:** <!-- How to split, or where splits are defined -->

---

## Evaluation Metric

**Metric name:** <!-- e.g. "FID", "val_bpb", "chamfer distance", "top-1 accuracy" -->

**Direction:** <!-- "lower is better" or "higher is better" -->

**How to compute:** <!-- Describe the evaluation procedure precisely.
If the paper provides an eval script or references a standard library, note it here.
-->

**Log format** (what the agent greps for in run.log):
```
# The agent will parse this exact pattern from run.log:
# Example: grep "^val_bpb:" run.log
# Example: grep "^fid:" run.log
```

---

## Paper's Reported Baseline

<!-- The number to beat / aim for. Agent uses this to gauge whether its implementation is in the right ballpark. -->

**Metric value:** <!-- e.g. val_bpb=0.997, FID=12.3 -->

**Model scale:** <!-- e.g. "50M params, trained for 100k steps on 8×A100" -->

**Notes:** <!-- Any caveats about the paper's reported number
Example: "Paper uses full nuScenes; we use a 10% subset, so expect ~2× worse metric"
-->

---

## Training Budget

**Time budget per run:** <!-- e.g. "5 minutes", "30 minutes" -->

**Hardware available:** <!-- e.g. "1× RTX 4090 (24GB VRAM)", "1× A100 (40GB)" -->

**Batch size guidance:** <!-- If paper specifies batch size and it's too large, note what to use instead -->

---

## Additional Constraints

<!-- Any other constraints the agent must respect:
- Packages available (pyproject.toml)
- Files that must not be modified
- Specific PyTorch version constraints
- Memory limits
-->

**Immutable files:** <!-- Files the agent must not modify, if any -->

**Available packages:** <!-- Point to pyproject.toml, or list key packages -->
