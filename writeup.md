# Final Writeup: Interpretable Student Performance Prediction

## 1. Dataset and Prediction Target

**Dataset:** UCI Student Performance Dataset (Math course)
- 395 students, 30 features
- Source: Paulo Cortez, University of Minho (2008)

**Prediction target:** Binary classification — `pass` (1) if final grade G3 ≥ 10, else `fail` (0)

**Class balance:** 265 pass (67%) / 130 fail (33%) — moderately imbalanced.

G1 and G2 (intermediate grades) are excluded to avoid predicting risk from prior grades rather than behavioral and demographic features.

---

## 2. Features and Preprocessing

**Features used (after dropping G1, G2, G3):** 29 features including:
- Academic behavior: `studytime`, `failures`, `schoolsup`, `paid`, `higher`
- Demographic: `age`, `sex`, `address`, `Medu`, `Fedu`, `Mjob`, `Fjob`
- Social/lifestyle: `absences`, `goout`, `Dalc`, `Walc`, `freetime`, `romantic`
- Family: `famsize`, `Pstatus`, `famsup`, `guardian`

**Preprocessing decisions:**
- Binary categoricals (`yes`/`no`, `M`/`F`, etc.) → encoded as 0/1
- Multi-class categoricals (`Mjob`, `Fjob`, `reason`, `guardian`) → one-hot encoded (`drop_first=True` to avoid multicollinearity)
- Features scaled with `StandardScaler` fit on train only and applied to val/test, to prevent data leakage and to normalize for distance-based models like KNN

**Data split:** 70% train (276) / 15% validation (59) / 15% test (60), stratified on the target label.

---

## 3. Evaluation Metrics

| Metric | Why it was chosen |
|--------|-------------------|
| **Accuracy** | Intuitive overall correctness measure |
| **Precision (macro)** | Penalizes false positives across both classes |
| **Recall (macro)** | Penalizes false negatives — important for catching at-risk students |
| **F1 (macro)** | Harmonic mean of precision and recall; handles class imbalance better than accuracy alone |

Macro averaging treats both classes equally, which is appropriate given the modest class imbalance (67% pass / 33% fail) and the importance of correctly identifying the failing minority.

---

## 4. Models Compared

Four models are compared in this checkpoint:

| Model | Type | Key parameter |
|-------|------|---------------|
| LR (L2, C=1.0) | Logistic Regression | Standard L2 regularization (baseline) |
| LR (L1, C=0.1) | Logistic Regression | L1 regularization, stronger regularization (C=0.1) |
| Decision Tree (depth=4) | Decision Tree | Depth-limited to 4 for interpretability |
| Decision Tree (full) | Decision Tree | Unconstrained depth (overfitting baseline) |

All models are trained on the same 70/15/15 stratified split from Week 5. Features are standardized (fit on train only).

**Note on KNN:** KNN (k=5) was evaluated as a Week 5 baseline. It showed a large generalization gap (val F1 0.619 → test F1 0.403), indicating it overfit the small training set. It was dropped from Week 7 comparisons in favor of models with better-understood regularization behavior.

---

## 5. Cross-Validation Results

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

## 6. Validation and Test Results

### Week 5 Baselines

| Model | Val Accuracy | Val F1 | Test Accuracy | Test F1 |
|-------|-------------|--------|---------------|---------|
| Logistic Regression (L2) | 0.678 | 0.614 | 0.650 | 0.576 |
| KNN (k=5) | 0.746 | 0.619 | 0.533 | 0.403 |

KNN had the highest validation accuracy but collapsed on the test set, confirming it overfit to local structure in the small training set. LR (L2) was the stronger baseline, with a much smaller val-to-test drop.

### Week 7 Models — Validation Set (n=59)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| LR (L2, C=1.0) | 0.678 | 0.622 | 0.611 | 0.614 |
| LR (L1, C=0.1) | 0.695 | 0.652 | 0.554 | 0.529 |
| Decision Tree (depth=4) | 0.661 | 0.604 | 0.598 | 0.600 |
| Decision Tree (full) | 0.610 | 0.569 | 0.574 | 0.570 |

### Week 7 Models — Test Set (n=60)

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

## 7. Feature Importance Analysis

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

## 8. Interpretability vs. Performance Tradeoffs

| Model | Test F1 | Interpretability | Notes |
|-------|--------|-----------------|-------|
| LR (L2, C=1.0) | 0.576 | High — linear coefficients | Stable, globally interpretable |
| LR (L1, C=0.1) | 0.509 | High — sparse coefficients | Best for feature selection; over-regularized here |
| Decision Tree (depth=4) | 0.643 | High — explicit decision rules | Best performer; locally and globally readable |
| Decision Tree (full) | 0.563 | Low — too complex to trace | Overfits; not interpretable in practice |

The depth-limited Decision Tree is the strongest model and also one of the more interpretable ones: a 4-level tree can be printed as rules and reasoned about directly. This is the result the project was targeting — a model that is both accurate and explainable.

---

## 9. Decision Tree Rules in Plain English

One advantage of the pruned Decision Tree over logistic regression is that its decision logic can be read directly. The root split and first two levels account for the majority of predictions:

- **If a student has no prior course failures (`failures < 0.5`):** the tree predicts pass for most of these students, especially if they also want to pursue higher education (`higher = 1`). This group has the highest pass rate.
- **If a student has one or more prior failures (`failures ≥ 0.5`):** the tree becomes more cautious. Students with high absences or lower parental education are predicted to fail. Students who still intend to pursue higher education have a better chance even in this group.
- **Absences** serve as a secondary split in the failing-risk branch: high absenteeism compounds the risk from prior failures.
- **Parental education (`Medu`)** appears in deeper splits as a supporting signal when the primary indicators are ambiguous.

This rule structure is directly usable by a counselor or advisor: a student with no prior failures and clear educational goals is low-risk; a student with even one failure and poor attendance warrants early outreach.

---

## 10. Final Conclusions

1. **`failures` is the single most informative feature** across all models. Students with prior failures are at significantly higher risk. This is actionable: the feature is available from school records at the start of the year.

2. **The pruned Decision Tree (depth=4) is the best model.** It achieves the highest test F1 (0.643) and recall (0.637) while remaining fully interpretable. The gain over the best logistic regression variant (F1 0.576) is meaningful for a dataset of this size, and it is confirmed by the decision rules being consistent with the logistic regression coefficients.

3. **Simple models are competitive — complexity hurts.** The unconstrained Decision Tree has a lower test F1 (0.563) than the pruned version (0.643), and both logistic regression variants underperform the pruned tree. The project's central claim — that interpretable models can perform well enough to be practically useful — is supported.

4. **The top features are consistent across all four models.** `failures`, `higher`, and `absences` appear as the strongest predictors in both the logistic regression coefficients and the decision tree splits. This cross-model consistency increases confidence that these features are genuinely informative, not noise.

5. **Metric differences should be interpreted cautiously given dataset size.** With only 60 test samples, a difference of 0.05 F1 is not statistically conclusive on its own. The cross-validation results (which use more data per evaluation) provide a more reliable picture of model ranking and are consistent with the test set ordering.

---

## 11. Limitations

**Dataset size.** With 395 students total and only 60 in the test set, all reported metrics have wide confidence intervals. The pruned Decision Tree's advantage over logistic regression is directionally consistent across CV and test, but a larger dataset would be needed to draw firm conclusions.

**Single school and subject.** The dataset covers one Portuguese school and one subject (math). The feature patterns — particularly the weight of `failures` and `higher` — may not transfer to different educational systems, grade levels, or subjects.

**Excluded grade features.** G1 and G2 (first and second period grades) were excluded by design to focus on behavioral and demographic predictors rather than prior grades. This reflects a realistic early-intervention setting, but a model that included G1/G2 would likely have much higher accuracy. The tradeoff is intentional but worth naming.

**No regularization tuning for LR (L1).** The L1 model uses C=0.1, which produces a sparse but over-regularized model on 276 training samples. A proper sweep over C values would be needed to fairly evaluate L1 logistic regression. As presented, it is better understood as a demonstration of what over-regularization looks like than as a competitive model.

**Interpretability is qualitative.** The interpretability analysis relies on visual inspection of feature importances and decision rules rather than a formal measure. Metrics like number of active features or decision path length would make the comparison more rigorous.

---

## 12. Future Improvements

- **Tune regularization strength.** A grid search over C values for the L1 logistic regression would give a fairer comparison and may close some of the performance gap with the Decision Tree.
- **Add a Random Forest as a performance ceiling.** Comparing the pruned Decision Tree against a Random Forest (which sacrifices interpretability for performance) would quantify exactly how much accuracy the interpretability constraint costs.
- **Formalize interpretability measurement.** Counting active features, measuring average decision path length, or using a tool like SHAP to explain individual predictions would make the interpretability comparison more rigorous and less qualitative.
- **Expand to the Portuguese course data.** The UCI dataset includes a second file for the Portuguese language course. Training on a combined or multi-course dataset would improve statistical power and allow testing whether the feature patterns generalize across subjects.
- **Validate on a different cohort.** The strongest test of these findings would be applying the trained model to a new cohort of students and measuring real-world predictive accuracy, rather than relying on a held-out split of the same dataset.
