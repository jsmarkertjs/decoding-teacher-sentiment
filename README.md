# Decoding Teacher Sentiment

A multi-task 1D CNN that predicts Quality and Difficulty scores from RateMyProfessors.com reviews, with LIME explainability showing which words drive each prediction.

> Built for DATA-441 (Applied Machine Learning) at American University.

---

## What It Does

RateMyProfessors scores are just averages. A 3.5 could mean "easy grader but bad teacher" or "hard grader but great teacher" — you can't tell from one number. This model separates those two factors: it reads the review text and scores it for quality (how good the teaching is) and difficulty (how hard the class is). Then LIME shows exactly which words pushed each score up or down.

| Task | Output | Best Validation MAE |
|---|---|---|
| **Quality** | 1.0-5.0 (teaching clarity) | **0.64** |
| **Difficulty** | 1.0-5.0 (how hard) | **0.78** |

### How It Works

1. **Scrape** -- Fetch reviews from RateMyProfessors.com via their GraphQL API
2. **Predict** -- A shared 1D CNN with two output heads predicts both scores at once
3. **Explain** -- LIME highlights the specific words that influenced each prediction

---

## Architecture

```
Input (150 tokens)
    |
Embedding (10,000 -> 100d)
    |
Conv1D (128 filters, kernel_size=5, ReLU)
    |
GlobalMaxPooling1D
    |
Dropout (0.5)
    |
+---------------------+---------------------+
| Dense(64 -> ReLU)   | Dense(64 -> ReLU)   |
| Dense(1 -> linear)  | Dense(1 -> linear)  |
| Quality Score       | Difficulty Score    |
+---------------------+---------------------+
```

**Total parameters:** 1,080,770

The model uses multi-task learning -- a shared representation learns general language patterns while separate heads learn what makes a review indicate quality vs. difficulty. This works because both tasks operate on the same input text.

---

## Example: LIME Explainability

**Review:** *"Great professor really explains concepts well and makes learning fun"*

**Quality explanation:**

| Word | Impact | Direction |
|---|---|---|
| great | +0.32 | Increases quality |
| explains | +0.28 | Increases quality |
| fun | +0.21 | Increases quality |
| well | +0.15 | Increases quality |
| professor | +0.08 | Increases quality |

**Difficulty explanation:**

| Word | Impact | Direction |
|---|---|---|
| explains | -0.25 | Decreases difficulty |
| great | -0.18 | Decreases difficulty |
| fun | -0.12 | Decreases difficulty |
| learning | -0.10 | Decreases difficulty |

Positive words like *great*, *explains*, and *fun* boost quality scores while lowering difficulty -- the model learns that well-explained material feels easier to students.

---

## Repository Structure

```
decoding-teacher-sentiment/
+-- README.md
+-- LICENSE                   MIT
+-- requirements.txt
+-- .gitignore
+-- notebooks/
|   +-- project_final.ipynb   Complete analysis notebook (cleaned)
+-- src/
|   +-- __init__.py
|   +-- scraper.py            GraphQL scraper (curl_cffi)
|   +-- preprocessing.py      Text cleaning, tokenization, padding
|   +-- model.py              Multi-task 1D CNN definition
|   +-- lime_explainer.py     LIME wrappers with 2D-array hack
+-- demo/
|   +-- app.py                Streamlit demo app
|   +-- example_data.json     Pre-computed LIME explanations
|   +-- requirements.txt      Demo-only dependencies
+-- docs/
|   +-- index.html            GitHub Pages site
|   +-- presentations/
|       +-- Decoding_Teacher_Sentiment.pdf
+-- data/
    +-- sample_reviews.csv    100 sample reviews
```

---


### Installation

```bash
git clone https://github.com/jsmarkertjs/decoding-teacher-sentiment.git
cd decoding-teacher-sentiment

python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook notebooks/project_final.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Scrape RateMyProfessors data (or load the sample CSV)
2. Preprocess and tokenize the text
3. Train the multi-task CNN
4. Show LIME explanations for sample reviews

### Run the Streamlit Demo

```bash
cd demo
pip install -r requirements.txt
streamlit run app.py
```

No TensorFlow needed for the demo -- it uses pre-computed LIME explanations.

---

## Limitations

- **Overfitting**: The model starts overfitting after about 3 epochs (training loss keeps dropping but validation loss rises). Early stopping or more regularization would help.
- **Data scope**: Trained on American University reviews only -- might not generalize to other schools.
- **No model weights in git**: The model is small enough (~4 MB) to retrain. Run the notebook to train fresh weights.
- **LIME hack**: The 2D-array trick for regression LIME works but is a workaround.

---

## No API Keys Required

The Gemini LLM branch has been removed entirely. No API keys, no rate limits, no external services. Everything runs locally.

---

## License

MIT 

---

## Acknowledgments

- Scraper based on [ratemyprofessor-api](https://github.com/ppannuta/ratemyprofessor-api) by ppannuta
- LIME: [Ribeiro et al., "Why Should I Trust You?" (KDD 2016)](https://arxiv.org/abs/1602.04938)
- Built with TensorFlow, scikit-learn, and Streamlit