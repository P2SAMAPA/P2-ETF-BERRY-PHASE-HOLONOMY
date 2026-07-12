# Berry Phase Holonomy for ETFs

Tracks the geometric (Berry) phase accumulated by a parameter vector driven cyclically through macro regimes. Nonzero holonomy signals path‑dependent, non‑integrable market structure – useful for detecting memory beyond Markovian models. The per‑ETF score is the Berry phase magnitude.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- State vector: mean, volatility, skewness, kurtosis
- Macro‑driven loop in parameter space
- Berry phase via solid angle / loop area
- Score = Berry phase magnitude (higher = stronger path‑dependence)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-berry-phase-holonomy-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High Berry phase → strong path‑dependent / non‑Markovian structure.
- Low Berry phase → Markovian / integrable structure.

## Requirements

See `requirements.txt`.
