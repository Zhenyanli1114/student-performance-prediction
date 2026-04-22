# Interpretable Student Performance Prediction

## Project Prompt

I would like to build a small, interpretable machine learning pipeline that predicts student academic outcomes based on behavioral and academic features such as study time, attendance, prior grades, and assignment patterns. The purpose of the project is not only to make predictions, but also to understand which features are most useful and which models provide the best balance of accuracy and explainability.

Because this is a course project rather than a production intervention system, I will use a public educational dataset and focus on a manageable binary or multiclass prediction task, such as predicting whether a student is at risk of low final performance. I will compare a few standard supervised learning methods covered by the course, such as logistic regression, k-nearest neighbors, decision trees, and possibly a simple ensemble if feasible.

The core question of the project is: **can simple and explainable models predict student outcomes well enough to support early intervention decisions, and what tradeoffs exist between model performance and interpretability?**

The final deliverable will be a reproducible Python project that:
1. cleans and explores the dataset,
2. trains several supervised learning models,
3. compares them using honest evaluation methods,
4. analyzes feature importance or decision logic,
5. presents conclusions about which models are most appropriate for a small educational prediction setting.

This project fits the course because it uses supervised learning, model evaluation, and tradeoff analysis, all of which are central themes of the syllabus.

## Execution Plan

### By Week 5 Homework Deadline

- Select a public student performance dataset and define a specific prediction target.
- Clean the data and perform exploratory data analysis.
- Build a train/validation/test workflow.
- Implement at least two simple baseline models, such as logistic regression and k-nearest neighbors.
- Choose evaluation metrics such as accuracy, precision/recall, or F1 depending on class balance.

This step matters because the course emphasizes honest evaluation and avoiding misleading model comparisons.

#### Week 5 Artifacts

By this checkpoint, I expect to have:
- cleaned dataset pipeline
- exploratory data analysis plots
- baseline model results
- a short writeup explaining the target, features, and metrics

### By Week 7 Homework Deadline

- Add stronger or more interpretable models such as decision trees or regularized logistic regression.
- Compare models using cross-validation or a comparable evaluation procedure.
- Analyze which features appear most informative.
- Create plots and tables comparing performance and interpretability.

This step matters because the syllabus highlights evaluation, overfitting control, and comparing methods rather than just building one model.

#### Week 7 Artifacts

By this checkpoint, I expect to have:
- 3 to 4 model comparisons
- metric summary tables
- interpretability analysis
- preliminary conclusions about tradeoffs

### By the Final Due Date

- Finalize code and documentation.
- Refine conclusions about which model is most suitable and why.
- Prepare final artifacts and README.
- Record a final video showing the project workflow, code organization, model comparisons, and tradeoffs.

The final video will demonstrate how the project works, show the code structure, and explain implementation tradeoffs, in line with the final project requirements.

#### Final Artifacts

By the final deadline, I expect to have:
- complete repo
- polished README
- final plots and result summaries
- short technical demo video
- explanation of tradeoffs, limitations, and future improvements

## Dataset

Cortez, P. & Silva, A. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

> P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." In A. Brito and J. Teixeira Eds., *Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008)*, pp. 5-12, Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.
