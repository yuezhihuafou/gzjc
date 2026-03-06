# Multi-Task Transfer Report (2026-03-06)

## Goal
- Stage A: pretrain backbone on XJTU (`sound_api_cache`, `task=hi`).
- Stage B: finetune on CWRU multi-head (`task=multi`, heads: `risk_binary + fault_type + condition_id`) with backbone-only init.
- Evaluate CWRU under both `split_mode=random` and `split_mode=leave_one_condition_out`.

## Code Snapshot
- Multi-head training pipeline added (`task=multi`).
- New flag `--init_backbone_only` to load only `backbone.pth` from init checkpoint.
- Added multi-head loss weights:
  - `--loss_w_risk` (default `1.0`)
  - `--loss_w_fault` (default `1.0`)
  - `--loss_w_condition` (default `0.5`)
- CWRU multi-task labels/data contract added in loader:
  - `risk_binary: label==0 -> 0, label>0 -> 1`
  - `fault_type: 0/1/2/3`
  - `condition_id: load+rpm -> class index`
- Added outputs for multi-task runs:
  - `overall_metrics.csv`
  - `risk_metrics.csv`
  - `fault_metrics.csv`
  - `condition_metrics.csv`
  - `per_condition_metrics.csv`
  - `risk_threshold.txt`
  - `risk_score_direction.txt`

## Experiment Outputs

### Stage A (XJTU HI pretrain, seed=42, epochs=3)
- Run: `/home/swy/gzjc/runs/20260306_multitask/stageA_xjtu_hi_pretrain_seed42_e3`
- `test_mae=0.1490`, `test_rmse=0.1958`

### Stage B (CWRU multi, random split, seed=42, epochs=3)
- Run: `/home/swy/gzjc/runs/20260306_multitask/stageB_cwru_multi_random_seed42_e3`
- Risk: `AUC=1.0000`, `BalancedAcc=1.0000`, `NormalRecall=1.0000`
- Fault: `Macro-F1=0.9882`
- Condition: `Macro-F1=0.7869`

### Stage B (CWRU multi, LOCO split, seed=42, epochs=3)
- Run: `/home/swy/gzjc/runs/20260306_multitask/stageB_cwru_multi_loco_seed42_e3`
- Risk: `AUC=1.0000`, `BalancedAcc=0.9989`, `NormalRecall=0.9979`
- Fault: `Macro-F1=0.8429`
- Condition: `Macro-F1=0.0000`

## Interpretation
- `risk/fault` generalization is strong under LOCO.
- `condition` in LOCO is open-set by construction (test condition unseen in train), so closed-set macro-F1 can collapse to `0`. This does not indicate `risk/fault` failure.

