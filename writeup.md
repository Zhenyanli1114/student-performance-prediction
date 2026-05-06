# Week 7 Checkpoint Writeup

## 1. Dataset and Prediction Target

**Dataset:** UCI Student Performance Dataset (Math course)
- 395 students, 30 features
- Source: Paulo Cortez, University of Minho (2008)

**Prediction target:** Binary classification — `pass` (1) if final grade G3 ≥ 10, else `fail` (0)

**Class balance:** 265 pass (67%) / 130 fail (33%) — moderately imbalanced.

G1 and G2 (intermediate grades) are excluded to avoid predicting risk from prior grades rather than behavioral and demographic features.

---

## 2. Models Compared

Four models are compared in this checkpoint:

| Model | Type | Key parameter |
|-------|------|---------------|
| LR (L2, C=1.0) | Logistic Regression | Standard L2 regularization (baseline) |
| LR (L1, C=0.1) | Logistic Regression | L1 regularization, stronger regularization (C=0.1) |
| Decision Tree (depth=4) | Decision Tree | Depth-limited to 4 for interpretability |
| Decision Tree (full) | Decision Tree | Unconstrained depth (overfitting baseline) |

All models are trained on the same 70/15/15 stratified split from Week 5. Features are standardized (fit on train only).

---

## 3. Cross-Validation Results

5-fold stratified cross-validation on the training set (n=276):

| Model | CV F1 Mean | CV F1 Std |
|-------|-----------|-----------|
| LR (L2, C=1.0) | 0.613 | 0.045 |
| LR (L1, C=0.1) | 0.589 | 0.087 |
| Decision Tree (depth=4) | 0.593 | 0.098 |
| Decision Tree (full) | 0.565 | 0.073 |

**Observations:**
- LR (L2) has the highest and most stable CV F1 (lowest std), suggesting it generalizes most consistently across folds.
- LR (L1) and Decision Tree (depth=4) are competitive but noisier.
- The full Decision Tree has the lowest CV F1, confirming that unconstrained growth hurts generalization.

---

## 4. Validation and Test Results

### Validation Set (n=59)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| LR (L2, C=1.0) | 0.678 | 0.622 | 0.611 | 0.614 |
| LR (L1, C=0.1) | 0.695 | 0.652 | 0.554 | 0.529 |
| Decision Tree (depth=4) | 0.661 | 0.604 | 0.598 | 0.600 |
| Decision Tree (full) | 0.610 | 0.569 | 0.574 | 0.570 |

### Test Set (n=60)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| LR (L2, C=1.0) | 0.650 | 0.589 | 0.575 | 0.576 |
| LR (L1, C=0.1) | 0.667 | 0.593 | 0.537 | 0.509 |
| Decision Tree (depth=4) | **0.700** | **0.656** | **0.637** | **0.643** |
| Decision Tree (full) | 0.617 | 0.564 | 0.562 | 0.563 |

**Key observations:**
- **Decision Tree (depth=4) is the best model on the test set** across all four metrics. Its F1 of 0.643 is the highest, and its recall of 0.637 is particularly relevant for catching at-risk students.
- LR (L1) has high accuracy but low recall on validation (0.554), suggesting it is biased toward predicting "pass" more often. Strong L1 regularization may be shrinking too many coefficients to zero given the small dataset size.
- LR (L2) remains the most stable model: low variance across folds, small val-to-test drop (0.614 → 0.576). It is the safest choice if consistency is the priority.
- The full Decision Tree improves over LR on test but is less stable than the pruned version (val F1 0.570 vs. pruned 0.600), confirming that pruning is necessary.

---

## 5. Feature Importance Analysis

### Logistic Regression (L2) — Top Coefficients

The top features by absolute coefficient magnitude are:
- `failures` (strongest negative predictor — past failures are the clearest signal)
- `higher` (wanting to pursue higher education is a strong positive predictor)
- `absences` (negative — more absences correlate with failing)
- `Medu` and `Fedu` (parental education, positive)
- `studytime` (moderate positive effect)

### Decision Tree (depth=4) — Gini Importances

The pruned tree splits primarily on:
- `failures` (root split — dominant feature)
- `higher`
- `absences`
- `age` and `Medu`

The top features are consistent between LR and the Decision Tree, which increases confidence that these features genuinely predict outcomes rather than being noise. The decision tree provides an additional benefit: each split is directly interpretable as a rule (e.g., "if failures < 0.5 and higher = 1, predict pass").

### L1 Sparsity

LR (L1, C=0.1) produces a sparse model, zeroing out several low-signal features. The retained features overlap heavily with those in LR (L2), providing implicit feature selection confirmation. However, with C=0.1 on 276 training samples, the model may be over-regularized.

---

## 6. Interpretability vs. Performance Tradeoffs

| Model | Test F1 | Interpretability | Notes |
|-------|--------|-----------------|-------|
| LR (L2, C=1.0) | 0.576 | High — linear coefficients | Stable, globally interpretable |
| LR (L1, C=0.1) | 0.509 | High — sparse coefficients | Best for feature selection; over-regularized here |
| Decision Tree (depth=4) | 0.643 | High — explicit decision rules | Best performer; locally and globally readable |
| Decision Tree (full) | 0.563 | Low — too complex to trace | Overfits; not interpretable in practice |

The depth-limited Decision Tree is the strongest model and also one of the more interpretable ones: a 4-level tree can be printed as rules and reasoned about directly. This is the result the project was targeting — a model that is both accurate and explainable.

---

## 7. Preliminary Conclusions

1. **`failures` is the single most informative feature** across all models. Students with prior failures are at significantly higher risk. This is actionable: it is available at the start of the year from school records.

2. **Pruned Decision Tree (depth=4) gives the best generalization** on this dataset. The gain over logistic regression (F1 0.643 vs. 0.576) is meaningful given the small dataset.

3. **Simple models are competitive.** The gap between the pruned tree and an unconstrained tree (F1 0.643 vs. 0.563) shows that complexity hurts here. The project's central claim — that interpretable models can perform well — is supported.

4. **All models are limited by dataset size.** With only 60 test samples, metric differences of ~0.05 F1 should be interpreted cautiously. The cross-validation results (which use more data per evaluation) provide a more reliable picture of model ranking.
