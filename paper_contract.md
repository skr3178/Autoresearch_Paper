# Paper Contract

<!-- The agent fills this in during Phase 1, before writing any code. -->
<!-- This is the ground truth for what is being implemented. If this is wrong, everything downstream is invalid. -->

## Dataset Contract

**Name:**
**Source:**
**Sample format:** <!-- type, shape, dtype, value range for each field -->
**Coordinate frame / units:** <!-- if spatial data -->
**Preprocessing:**
**Split sizes:** <!-- train: N, val: M, test: K -->

## Architecture Contract

<!-- Every module the paper defines, with input → output shapes in paper notation -->

| Module | Input shape | Output shape | Notes |
|--------|-------------|--------------|-------|
| | | | |

## Training Contract

**Optimizer:**
**Learning rate schedule:**
**Batch size:**
**Steps / epochs:**

### Loss Mapping

| Paper symbol | Code variable | Semantic meaning | Value |
|-------------|---------------|------------------|-------|
| | | | |

## Evaluation Contract

**Metric name:**
**Direction:** <!-- lower is better / higher is better -->
**Single-sample or best-of-K?** <!-- If best-of-K, what K? -->
**GT source:** <!-- raw data or reconstructed? -->
**Normalization:**
**Thresholds / filtering:**
**Sampling count / seed policy:**
**Paper-specific quirks:** <!-- any non-standard evaluation details -->
**Expected value:** <!-- paper's reported number -->
**Tolerance:** <!-- acceptable deviation given hardware/data differences -->

## Inference Contract

**Sampling procedure:**
**Number of steps:**
**Schedule:** <!-- must match training endpoints -->
**Post-processing:**

## Ambiguity Register

<!-- Every detail the paper leaves unclear -->

| # | Paper section | Detail | Impact | Default chosen | Rationale | Alternative |
|---|---------------|--------|--------|----------------|-----------|-------------|
| | | | low/med/high | | | |
