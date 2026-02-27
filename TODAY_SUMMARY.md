# Today Summary (2026-02-26)

## Key Outcomes
- Evaluated three risk checkpoints on CWRU with binary alignment: Normal=0, B/IR/OR=1.
- New 0226 checkpoint (risk_index_small_0226) matches 0212 performance; horizon10_50ep performs poorly on CWRU.
- CWRU split is random (seed=42) and highly imbalanced in binary form (Normal=1652, Fault=33257 overall).

## CWRU Results (binary alignment)
- risk_index_small_0212:
  - Test Loss 0.4726, Acc 0.8301, AUC 0.5745, PR-AUC 0.9757
  - Confusion: [[10, 207], [683, 4337]]
- risk_index_small_horizon10_50ep:
  - Test Loss 3.1466, Acc 0.5534, AUC 0.1488, PR-AUC 0.9101
  - Confusion: [[2, 215], [2124, 2896]]
- risk_index_small_0226:
  - Test Loss 0.4726, Acc 0.8301, AUC 0.5745, PR-AUC 0.9757
  - Confusion: [[10, 207], [683, 4337]]

## Data Alignment Changes
- Created binary-aligned CWRU labels for risk task:
  - datasets/cwru/cwru_processed_risk/labels.npy
  - signals.npy hardlinked to datasets/cwru/cwru_processed/signals.npy
- Created junction so code can read expected path:
  - cwru_processed -> datasets/cwru/cwru_processed_risk

## Code Change
- dl/data_loader.py now accepts single-channel CWRU signals and expands to 2 channels:
  - (N, L) or (N, 1, L) -> (N, 2, L)

## Notes
- experiments/train.py eval_only for risk + cwru is placeholder; direct evaluation used instead.
- XJTU risk task uses time-based horizon; CWRU binary alignment is a different task, so cross-domain AUC is low.

## Next Possible Steps
- If goal is cross-condition robustness, add load/rpm stratified splits or leave-one-condition-out evaluation.
- Define CWRU risk labels by load/rpm if needed (awaiting rule definition).
