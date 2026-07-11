# Decoding Teacher Sentiment

Multi-task 1D CNN predicting Quality & Difficulty from RateMyProfessor reviews, with LIME explainability.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/project_final.ipynb
```

## Repo Structure

- `notebooks/project_final.ipynb` -- Complete cleaned notebook (no API keys, no Gemini, no Colab paths)
- `src/` -- Modular Python scripts extracted from the notebook
  - `scraper.py` -- RateMyProf GraphQL API client (curl_cffi)
  - `preprocessing.py` -- clean_for_nn(), tokenizer, padding, train/test split
  - `model.py` -- build_model(): multi-task 1D CNN definition
  - `lime_explainer.py` -- LIME wrappers with "2D array hack" for regression
- `demo/` -- Streamlit app with pre-computed LIME explanations (no TF needed)
- `docs/` -- GitHub Pages site + presentation PDF
- `data/sample_reviews.csv` -- 100 sample reviews

## Architecture

Embedding(10000->100d) -> Conv1D(128, k=5) -> GlobalMaxPool -> Dropout(0.5) -> [Dense(64)->Dense(1)] x 2 heads

1,080,770 parameters. Trained on 27,874 reviews from American University. Quality MAE ~0.54, Difficulty MAE ~0.42 at best epoch.

## Key Conventions

- No API keys, no external services, no Gemini. Everything runs locally.
- No emojis in any file.
- README and comments should be human/natural, not corporate.
- The "2D array hack" in LIME wrappers duplicates continuous scores into 2 columns for LIME's classification API.
- Model overfits after ~3 epochs -- early stopping recommended.
- No model weights tracked in git (~4 MB, just retrain).
- docs/assets/ directory intentionally omitted -- no placeholder images.

## Build Artifacts

- `docs/index.html` -- GitHub Pages site (light/dark mode, responsive)
- `demo/app.py` -- Streamlit demo (run with `cd demo && pip install -r requirements.txt && streamlit run app.py`)
- `notebooks/project_final.ipynb` -- Full analysis notebook

## Remaining Setup

- Update `jsmarkertjs/decoding-teacher-sentiment` URLs in README.md, docs/index.html, demo/app.py to match actual GitHub repo
- Enable GitHub Pages from docs/ folder in repo settings
- Initialize git: `git init && git add . && git commit -m "Initial commit"`