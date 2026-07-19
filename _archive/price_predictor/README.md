# Archived: AI Stock Predictor (price-based)

This folder holds the **previous** product stack that lived at the repo root:

- Technical-feature data pipeline (`data.py`, yfinance OHLCV)
- PyTorch / XGBoost / RF price models (`model.py`)
- Offline global training (`train_global.py`, `global_models/`)
- Signal journal (`journal.py`)
- PBKDF2 login UI branded “AI Stock Predictor” (`auth.py`)

The **live** app is the Fundamental Stock Analyzer:

- Entry point: `app.py`
- Library: `src/`
- Auth: `src/auth.py`

Nothing under `_archive/` is imported by the running app. Kept for reference only;
safe to delete once you no longer need the price-predictor code.
