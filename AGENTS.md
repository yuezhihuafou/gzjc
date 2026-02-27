# Repository Guidelines

## Project Structure & Module Organization
- `dl/`: core deep-learning code (data loaders, model backbone, losses).
- `experiments/`: training/inference entry points (for example `train.py`, `inference.py`) and output artifacts under `experiments/outputs/`.
- `tools/`: data preparation and utility scripts (`load_cwru.py`, `load_xjtu.py`, `build_sound_api_cache.py`, `tools/sound_api/*`).
- `core/`: traditional feature/model pipeline used by classic ML experiments.
- `datasets/`: local data storage (raw + processed). Large files should stay local and out of commits.
- `docs/`: operational and architecture documentation; use this for design notes and process docs.
- `legacy/`: historical/reference scripts, not the main training path.

## Build, Test, and Development Commands
- Create env (Linux): `bash setup_conda_env.sh`
- Activate env (Linux): `source activate_env.sh`
- Install pip deps directly: `pip install -r requirements.txt`
- Prepare CWRU data: `python tools/load_cwru.py --data_dir <raw_dir> --output_dir cwru_processed`
- Train (CWRU): `python experiments/train.py --data_source cwru --batch_size 32 --epochs 100`
- Train (risk task): `python experiments/train.py --data_source sound_api_cache --task risk --horizon 10 --batch_size 8 --epochs 50`
- Inference: `python experiments/inference.py --data_source cwru --threshold 0.4`

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indentation, UTF-8 source files.
- Use `snake_case` for files/functions/variables, `PascalCase` for classes, and clear CLI flags via `argparse`.
- Keep modules focused: loaders in `dl/`, one-off processing in `tools/`, experiments in `experiments/`.
- Prefer small, composable functions and explicit tensor shape comments where non-obvious.

## Testing Guidelines
- No unified `pytest` suite exists yet; validate changes with targeted script runs.
- For data pipeline checks, run scripts such as:
  - `python tools/verify_cache_labels.py`
  - `python tools/sound_api/verify_mc_pipeline.py --mc_dir <dir>`
- When changing training logic, run at least one short smoke train (1-3 epochs) and record key metrics (loss/acc/AUC) in PR notes.

## Commit & Pull Request Guidelines
- Follow concise conventional-style commits seen in history (`init`, `chore: ...`, `Initial commit ...`).
- Recommended format: `<type>: <short summary>` (for example `feat: add cwru risk split by load`).
- PRs should include:
  - scope and motivation,
  - commands executed,
  - metric deltas or sample outputs,
  - any dataset/checkpoint impact (size, path, reproducibility notes).
- Do not commit secrets, API keys, or large generated artifacts unless explicitly required.
