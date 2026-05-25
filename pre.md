# Video Script (~13–14 minutes)

---

## Part 1 — Introduction (1.5 min)

> "Hi, this is my final project for the course — Interpretable Student Performance Prediction. The core question I'm trying to answer is: can simple, explainable machine learning models predict whether a student will pass or fail well enough to actually be useful for early intervention? Not just accurate, but understandable enough that a school counselor could look at the output and act on it.
>
> I'm using the UCI Student Performance dataset — 395 students from a Portuguese school, 30 behavioral and demographic features. The prediction target is binary: pass if the final grade G3 is 10 or above, fail otherwise.
>
> I'll show you the demo first, then walk through the code, then talk about the tradeoffs I had to make."

---

## Part 2 — Demo (3.5 min)

**Open terminal. Navigate to the project folder.**

> "The pipeline runs as four sequential scripts. Let me walk through them."

**Run: `python data_pipeline.py`**

> "The first script downloads the raw dataset from UCI, cleans it, encodes the categorical features, and produces three CSV files — train, validation, and test — using a stratified 70/15/15 split. You can see the class balance printed here: 265 pass, 130 fail. Moderately imbalanced, which is why I use macro F1 rather than accuracy as the primary metric."

**Open `data/` folder briefly to show train.csv, val.csv, test.csv.**

**Run: `python eda.py`**

> "The second script generates the exploratory data analysis plots — class distribution, feature distributions, a correlation heatmap, and boxplots by outcome. These went into the writeup and helped identify which features looked informative before any modeling."

**Open `plots/` folder. Show `class_distribution.png`, `correlation_heatmap.png`, and `boxplots_by_outcome.png` briefly.**

**Run: `python baseline_models.py`**

> "Third script runs the Week 5 baselines — logistic regression and KNN. You can see the results printed here. LR gets a test F1 of 0.576. KNN gets 0.619 on validation but drops to 0.403 on test — a big generalization gap. That told me KNN was memorizing local structure rather than learning anything useful, so I dropped it for the final comparison."

**Run: `python week7_models.py`**

> "The final script runs all four Week 7 models — two logistic regression variants and two decision trees — does 5-fold cross-validation, evaluates on validation and test, and generates all the feature importance and confusion matrix plots. Here are the test results. The pruned Decision Tree with depth 4 is the best model — test F1 of 0.643, the highest recall at 0.637. That recall matters here because recall is what tells you how many at-risk students you're actually catching."

**Open `plots/` folder. Show `cv_comparison.png`, `val_vs_test_comparison.png`, `decision_tree_structure.png`, `fi_dt_pruned.png`.**

> "The decision tree structure plot is the most useful artifact for interpretability — you can actually read the rules it learned. The root split is on `failures`: students with no prior failures go left, and most of them pass. Students with failures go right, and from there absences and parental education become the deciding factors."

---

## Part 3 — Code Walkthrough (5 min)

**Open `data_pipeline.py`.**

> "The project is organized as four scripts, each doing one thing. `data_pipeline.py` handles everything up to the split. The main decisions here are in the encoding: binary categoricals like yes/no and M/F get mapped to 0 and 1 directly. Multi-class categoricals — job types, reason for choosing the school, guardian — get one-hot encoded with `drop_first=True` to avoid multicollinearity. The scaler is fit only on the training set and then applied to val and test — that's important to prevent data leakage."

**Open `eda.py` briefly.**

> "EDA is its own script so it's easy to re-run independently. Nothing unusual here — it reads the cleaned splits and produces the plots."

**Open `baseline_models.py`.**

> "The baseline script follows the same pattern as the final model script — load splits, scale, train, evaluate, plot. I kept KNN in this file as a record of what the Week 5 baselines looked like, even though it was dropped from the final comparison."

**Open `week7_models.py`.**

> "The main modeling script. A few things worth pointing out. The `cross_validate_models` function runs 5-fold stratified CV on the training set — stratified so each fold has the same class balance. I use `f1_macro` as the scoring metric here, which is consistent with the held-out evaluation. The `evaluate` function uses macro averaging for the same reason — it treats both classes equally rather than weighting by frequency.

> For the L1 logistic regression, I set `l1_ratio=1` and `solver='saga'` explicitly. This matters because newer versions of scikit-learn deprecated the `penalty` parameter, and without the explicit ratio the model silently ran as L2 — which would have made the L1 results meaningless. Catching that bug was one of the more important moments in the project."

**Scroll to the `main()` function.**

> "The main function runs CV first, then fits each model on the full training set for the held-out evaluation. Cross-validation uses scaled training data; the final evaluation uses the scaler fit once on training and applied to val and test. The plots all get saved to the `plots/` directory."

**Open `writeup.md`.**

> "The writeup covers the full analysis — dataset, preprocessing decisions, model comparisons, feature importance, interpretability discussion, decision tree rules in plain English, and the conclusions. I'll reference it during the tradeoffs section."

---

## Part 4 — Tradeoffs (4 min)

**Keep `writeup.md` open for reference.**

> "Let me talk through the key decisions I had to make and why I made them."

> **"Binary threshold at G3 ≥ 10."** The dataset has a continuous final grade from 0 to 20. I converted it to binary because the project goal is early intervention — a counselor needs a yes/no signal, not a predicted number. 10 is a natural passing boundary for the Portuguese grading system. The tradeoff is that I'm throwing away nuance. A student who gets a 9 versus a 15 look the same to the model, which isn't ideal, but it makes the prediction task actionable.

> **"Dropping G1 and G2."** The first and second period grades are extremely predictive of the final grade — if you include them, accuracy goes up a lot, but you're essentially predicting a grade from prior grades. That's not useful for early intervention, because you don't have those grades at the start of the year. I excluded them so the model has to rely on behavioral and demographic features that are available before any grades exist.

> **"Macro averaging over weighted averaging."** With a 67/33 class imbalance, weighted F1 would favor performance on the majority class. Macro treats pass and fail equally, which is right for this problem — catching a student who is going to fail is at least as important as correctly predicting a pass.

> **"Depth 4 for the Decision Tree."** I tried an unconstrained tree as a comparison. It had lower CV F1 and lower test F1 than the pruned version, which is the classic overfitting pattern on a small dataset. Depth 4 was a judgment call — deep enough to capture real interactions between features, shallow enough to remain readable and not overfit. The structure plot shows it's a tree a human can actually trace.

> **"C=0.1 for L1 regularization."** This turned out to be too strong. The model achieved decent accuracy by predicting pass most of the time, but its recall dropped to 0.537 on the test set — it was missing at-risk students. On 276 training samples, C=0.1 is aggressive. The right fix would be a grid search over C values, but I kept this model in the comparison as a concrete example of what over-regularization looks like, since that's a concept the course covers directly.

> **"5-fold CV over a single validation split."** The validation set is 59 samples. A single evaluation on 59 samples is noisy — one unlucky split could give a misleading picture of model quality. 5-fold CV on the training set gives five separate evaluations and lets me measure variance across folds. The CV results and the test set results are consistent in their model ranking, which increases confidence in the conclusions."

---

## Closing (30 sec)

> "To summarize: the pruned Decision Tree is the best model on this dataset — highest test F1, highest recall, and fully interpretable as a set of rules. The most important features are prior failures, intent to pursue higher education, and absences, and they're consistent across all four models. Simple models win here. Thanks."
