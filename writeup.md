# Week 5 Checkpoint Writeup

## 1. Dataset and Prediction Target

**Dataset:** UCI Student Performance Dataset (Math course)
- 395 students, 30 features
- Source: Paulo Cortez, University of Minho (2008)

**Prediction target:** Binary classification — `pass` (1) if final grade G3 ≥ 10, else `fail` (0)

This threshold reflects a common passing grade boundary and converts the task from regression to classification, making it actionable for early intervention decisions (the core goal of the project).

**Class balance:** 265 pass (67%) / 130 fail (33%) — moderately imbalanced.

**Why drop G1 and G2?** G1 and G2 are first- and second-period grades. Including them would essentially predict the final grade from prior grades, bypassing the behavioral and demographic features the project is interested in. The goal is to predict risk from observable inputs *before* grades are available.

---

## 2. Features and Preprocessing

**Features used (after dropping G1, G2, G3):** 29 features including:
- Academic behavior: `studytime`, `failures`, `schoolsup`, `paid`, `higher`
- Demographic: `age`, `sex`, `address`, `Medu`, `Fedu`, `Mjob`, `Fjob`
- Social/lifestyle: `absences`, `goout`, `Dalc`, `Walc`, `freetime`, `romantic`
- Family: `famsize`, `Pstatus`, `famsup`, `guardian`

**Preprocessing decisions:**
- Binary categoricals (`yes`/`no`, `M`/`F`, etc.) → encoded as 0/1
- Multi-class categoricals (`Mjob`, `Fjob`, `reason`, `guardian`) → one-hot encoded (drop_first=True to avoid multicollinearity)
- Features scaled with `StandardScaler` (fit on train only, applied to val/test) to normalize for distance-based models like KNN

**Data split:** 70% train (276) / 15% validation (59) / 15% test (60), stratified on target.

---

## 3. Evaluation Metrics

| Metric | Why it was chosen |
|--------|-------------------|
| **Accuracy** | Intuitive overall correctness measure |
| **Precision (macro)** | Penalizes false positives across both classes |
| **Recall (macro)** | Penalizes false negatives — important for catching at-risk students |
| **F1 (macro)** | Harmonic mean of precision and recall; handles class imbalance better than accuracy alone |

Macro averaging treats both classes equally, which is appropriate given the modest class imbalance and the importance of correctly identifying failing students.

---

## 4. Baseline Model Results

### Validation Set

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| Logistic Regression | 0.678    | 0.622     | 0.611  | 0.614 |
| KNN (k=5)           | 0.746    | 0.785     | 0.619  | 0.619 |

### Test Set

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| Logistic Regression | 0.650    | 0.589     | 0.575  | 0.576 |
| KNN (k=5)           | 0.533    | 0.396     | 0.425  | 0.403 |

**Observations:**
- KNN shows a large drop from validation to test (accuracy 0.746 → 0.533), indicating overfitting to the small training set. KNN with k=5 is sensitive to the local structure of the data and does not generalize well here.
- Logistic Regression is more stable across val and test (F1: 0.614 → 0.576), making it the stronger baseline despite lower validation accuracy.
- Both models are limited by the relatively small dataset (395 samples) and the absence of grade-based features. The Week 7 models (decision trees, regularized logistic regression) will aim to improve generalization.
