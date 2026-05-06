# Week 7 Reflection

## What I have achieved so far

For the Week 7 checkpoint, I added two new models (Decision Tree with depth=4 and an unconstrained Decision Tree) and a regularized Logistic Regression variant (L1, C=0.1), bringing the total to four models. I set up 5-fold stratified cross-validation on the training set to give a more reliable model comparison than a single val/test split. I generated feature importance plots for all four models and a visualization of the pruned decision tree's structure. I also wrote a full analysis of the interpretability vs. performance tradeoffs, which is the core question of the project.

## What I am happy with

The most satisfying result is that the depth-limited Decision Tree outperformed both logistic regression variants on the test set (F1 0.643 vs. 0.576) while remaining interpretable. This directly supports the project's main claim that simple, explainable models can perform well enough to be useful in an early intervention context. The fact that the top features — `failures`, `higher`, `absences` — are consistent across all four models increases confidence in those findings.

I'm also glad I caught a silent bug in the L1 logistic regression setup. The version of scikit-learn being used (1.8+) deprecated the `penalty` parameter, and using `penalty='l1'` without also setting `l1_ratio=1` caused the model to silently run as L2 with a different C. Fixing this changed the L1 results meaningfully, so it mattered for the analysis.

The decision to use 5-fold CV on the training set rather than relying solely on the val split was the right call — the val set (n=59) is too small to trust a single evaluation.

## What I am struggling with

The small dataset continues to be the main challenge. Even with cross-validation, the folds are only ~55 samples for validation per split, which leads to high variance in per-fold scores (CV std of 0.087 for LR L1 is high). I'm uncertain whether the Decision Tree's test advantage over logistic regression is a real effect or just favorable test-set random variation.

The L1 model with C=0.1 may be over-regularized for 276 training samples — it achieves high accuracy by biasing toward "pass" predictions, hurting recall. I haven't done a proper sweep over C values. For the final checkpoint I should either tune this more carefully or drop the L1 model in favor of something more clearly differentiated.

The interpretability analysis is mostly qualitative right now. I'm comparing feature importances visually and listing top features, but I don't have a rigorous measure of "how interpretable" each model is. I'm planning to formalize this for the final deliverable, possibly by counting decision paths or measuring the number of active features.

## What I'm planning for the final checkpoint

- Tune regularization strength more carefully, at minimum checking a few C values for the L1 model.
- Add a short discussion of what the decision tree rules say in plain English, making the interpretability argument more concrete.
- Consider whether to add one more comparison (e.g., a Random Forest) as a performance ceiling to show how much interpretable models give up.
- Finalize code documentation and repo structure for the submission.
- Record the final demo video walkthrough.

## Feedback I'd like from course staff

1. **Decision Tree vs. LR gap** — The pruned Decision Tree has a notably higher test F1 (0.643 vs. 0.576) than LR. Is this difference large enough to draw conclusions on a dataset of this size, or should I treat it more cautiously?

2. **Interpretability operationalization** — Is listing feature importances and decision rules sufficient to claim a model is "interpretable," or is there a more rigorous standard I should be meeting for the final deliverable?

3. **L1 model** — Given that LR (L1, C=0.1) underperforms, is it worth keeping in the final comparison as a demonstration of what over-regularization looks like, or does it weaken the overall narrative?
