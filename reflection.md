# Week 5 Reflection

## What I have achieved so far

For the Week 5 checkpoint, I set up the full foundation of the project end-to-end. I found and downloaded a real dataset (UCI Student Performance, Math course), cleaned and preprocessed it, and made deliberate decisions about what the prediction target should be and which features to include. I built a reproducible data pipeline that handles encoding, scaling, and stratified train/val/test splitting. I ran exploratory data analysis and generated plots that helped me understand the class balance, the distribution of key features, and how those features relate to student outcomes. Finally, I implemented and evaluated two baseline models — logistic regression and KNN — and wrote up the results with proper evaluation metrics.

## What I am happy with

I'm happy with the decision to frame this as binary classification (pass/fail) rather than regression on the raw grade. It makes the task more actionable and directly relevant to the early intervention framing in my project proposal. I'm also glad I thought through the feature leakage issue with G1 and G2 early — dropping them was the right call and I think it makes the project more honest about what it's actually predicting.

The EDA plots came out clean and readable, and the results table in the writeup ended up being a genuinely interesting finding: KNN looked better on validation but collapsed on the test set, while logistic regression was more stable. That kind of result is more interesting to analyze than if both models had performed the same way.

## What I am struggling with / challenges ahead

The dataset is small (395 students), which makes it hard to draw confident conclusions — the val and test sets are only ~59 and 60 samples each, so metrics can swing a lot with a few predictions. I'm a bit worried about whether my results will be reliable enough to make strong claims about model comparisons in Week 7.

I'm also unsure about how much feature engineering I should do. Right now I'm using the raw features with minimal transformation. For Week 7, I'm considering things like interaction terms or binning absences, but I don't want to overfit to this specific dataset either.

The interpretability angle of the project (the core research question) is something I haven't tackled yet — I've just been focused on getting the pipeline working. I need to think more carefully about how to actually measure and compare interpretability across models, not just performance.

## Feedback I'd like from course staff

I'd especially appreciate feedback on:

1. **Feature selection** — Is it appropriate to keep all 29 features, or should I be doing some form of selection before modeling? I want to avoid the models picking up noise, especially with such a small dataset.

2. **The leakage decision** — I dropped G1 and G2 to avoid predicting grades from grades. Does that framing make sense, or is there a case for including them given that the goal is early intervention (where some grade data might realistically be available)?

3. **Interpretability measurement** — For Week 7, I plan to compare decision trees and logistic regression on interpretability. Is there a recommended way to operationalize "interpretability" beyond just looking at feature importances? I want the comparison to feel rigorous and not just qualitative.
