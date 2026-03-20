# Proof Log

## Submodule 1: data_loader

### 4a. Equation citations
No explicit equations for BEV rasterization / coordinate transforms were extracted into `paper/carplanner_equations.md` for this implementation stage.

⚠️ INVENTION: BEV raster is currently a zero tensor placeholder (no rasterization). Justification: Phase 3 submodule gate only checks shape/dtype/range/center and loader throughput; full rasterization requires additional paper details and map API integration.

### 4b. Algorithm citations
No algorithm blocks mapped to data_loader in `paper/algorithms.md` for this stage.

### 4c. Hyperparameter citations
- bev_h=224, bev_w=224, bev_c=7, resolution=0.5m/px, t_hist=5, t_future=8 → ⚠️ INVENTION (not traced to `paper/hyperparameters.md` in this stage). Chosen to satisfy `implementation/test_data_loader.py` expectations and provide a stable debug interface.

### 4d. Correctness evidence
- `implementation/test_data_loader.py` passes:
  - Shapes: bev (7,224,224), ego_history (5,3), gt_trajectory (8,3)
  - Dtypes: float32
  - Range: bev in [0,1]
  - Coordinate sanity: gt_trajectory is in global frame (large coordinates)
  - Determinism: dataset[0] twice returns identical tensors
  - Throughput: 10 samples in 2.91s (<5s for 10 batches at B=2 is not directly measured; sample throughput is acceptable for debug)

### 5. Figure verification (architecture figures only)
Runner preloaded figures; no additional figure reads performed this session.

Figure-to-code mapping (data pipeline components):
- Figure component "nuPlan logs / scenario" → `implementation/data_loader.py:NuPlanDataset` (loads scenarios via `NuPlanScenarioBuilder`)
- Figure component "BEV raster" → `implementation/data_loader.py:NuPlanDataset.__getitem__` (currently returns zero BEV; ⚠️ placeholder)
- Figure component "ego history" → `implementation/data_loader.py:NuPlanDataset.__getitem__` (currently repeats current ego pose; ⚠️ placeholder)
- Figure component "future trajectory GT" → `implementation/data_loader.py:NuPlanDataset.__getitem__` (uses `scenario.get_ego_future_trajectory`)

Reverse mapping:
- `implementation/data_loader.py:NuPlanDataset` → Figure component "data loader / scenario builder"
- `implementation/data_loader.py:collate_fn` → Figure component "batching"
- `implementation/data_loader.py:build_dataloader` → Figure component "dataloader"

⚠️ NOT FOUND / gaps:
- Proper BEV rasterization (map layers, agents, ego history) is not implemented yet.
- Ego history is not true history (placeholder repeats current state).

Status: Gate tests pass, but rasterization/history are placeholders to be filled when paper details are integrated.
