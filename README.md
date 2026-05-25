# Interpretable Student Performance Prediction

Binary classification project predicting whether a student will pass or fail their math course using behavioral and demographic features. The central question: **can simple, interpretable models perform well enough to support early academic intervention decisions?**

Dataset: UCI Student Performance Dataset (Math course) — 395 students, 30 features, sourced from Paulo Cortez, University of Minho (2008).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the scripts in order:

```bash
python data_pipeline.py   # download data, clean, encode, split into train/val/test
python eda.py             # exploratory data analysis plots
python baseline_models.py # logistic regression and KNN baselines
python week7_models.py    # model comparisons, cross-validation, feature importance
```

All plots are saved to `plots/`.

---

## File Structure

```
data_pipeline.py     — downloads and cleans the UCI dataset, produces train/val/test splits
eda.py               — EDA plots (class distribution, feature distributions, correlation heatmap)
baseline_models.py   — Week 5 baselines: Logistic Regression (L2) and KNN
week7_models.py      — Week 7 models: LR (L2), LR (L1), Decision Tree (depth=4), Decision Tree (full)
                       includes 5-fold CV, feature importance plots, confusion matrices
data/                — raw CSV and train/val/test splits (70/15/15 stratified split)
plots/               — all generated figures
writeup.md           — full analysis writeup with results, interpretability discussion, and conclusions
```

---

## Key Results

Four models were compared using 5-fold stratified cross-validation and a held-out test set (n=60):

| Model | CV F1 | Test F1 |
|-------|-------|---------|
| LR (L2, C=1.0) | 0.613 | 0.576 |
| LR (L1, C=0.1) | 0.589 | 0.509 |
| Decision Tree (depth=4) | 0.593 | **0.643** |
| Decision Tree (full) | 0.565 | 0.563 |

**The pruned Decision Tree (depth=4) is the best model**, achieving the highest test F1 while remaining fully interpretable as a set of human-readable rules. The most informative features across all models are `failures` (prior course failures), `higher` (intent to pursue higher education), and `absences`.

See [writeup.md](writeup.md) for the full analysis.
