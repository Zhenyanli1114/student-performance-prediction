import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

matplotlib.use("Agg")

DATA_DIR = "data"
PLOTS_DIR = "plots"


def load_splits():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train = train.drop(columns=["pass"])
    y_train = train["pass"]
    X_val = val.drop(columns=["pass"])
    y_val = val["pass"]
    X_test = test.drop(columns=["pass"])
    y_test = test["pass"]
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s


def evaluate(name, model, X, y):
    preds = model.predict(X)
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y, preds), 3),
        "Precision": round(precision_score(y, preds, average="macro", zero_division=0), 3),
        "Recall": round(recall_score(y, preds, average="macro", zero_division=0), 3),
        "F1": round(f1_score(y, preds, average="macro", zero_division=0), 3),
    }


def cross_validate_models(named_models, X_train, y_train, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for name, model in named_models:
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro")
        results.append({
            "Model": name,
            "CV F1 Mean": round(scores.mean(), 3),
            "CV F1 Std": round(scores.std(), 3),
        })
    return pd.DataFrame(results).set_index("Model")


def plot_cv_comparison(cv_df):
    _, ax = plt.subplots(figsize=(9, 5))
    models = cv_df.index.tolist()
    means = cv_df["CV F1 Mean"].values
    stds = cv_df["CV F1 Std"].values
    colors = ["#4a90d9", "#357abd", "#e67e22", "#c0392b"]

    bars = ax.bar(models, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("5-Fold Cross-Validation F1 Comparison (Train Set)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cv_comparison.png"), dpi=150)
    plt.close()
    print("Saved: plots/cv_comparison.png")


def plot_metric_comparison(val_df, test_df):
    metrics = ["Accuracy", "F1"]
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(val_df))
        width = 0.35
        ax.bar(x - width / 2, val_df[metric], width, label="Validation",
               color="#4a90d9", edgecolor="white")
        ax.bar(x + width / 2, test_df[metric], width, label="Test",
               color="#e67e22", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(val_df.index, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: Validation vs Test")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Model Comparison: Validation vs Test Performance", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_vs_test_comparison.png"), dpi=150)
    plt.close()
    print("Saved: plots/val_vs_test_comparison.png")


def plot_lr_feature_importance(model, feature_names, title, filename):
    coefs = np.abs(model.coef_[0])
    series = pd.Series(coefs, index=feature_names).sort_values(ascending=False).head(15)

    _, ax = plt.subplots(figsize=(8, 6))
    series[::-1].plot(kind="barh", ax=ax, color="#4a90d9")
    ax.set_xlabel("|Coefficient|")
    ax.set_title(f"Feature Importance — {title}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


def plot_dt_feature_importance(model, feature_names, title, filename):
    series = pd.Series(model.feature_importances_, index=feature_names)
    series = series[series > 0].sort_values(ascending=False).head(15)

    _, ax = plt.subplots(figsize=(8, 6))
    series[::-1].plot(kind="barh", ax=ax, color="#e67e22")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Feature Importance — {title}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


def plot_decision_tree_structure(model, feature_names, filename):
    _, ax = plt.subplots(figsize=(22, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Fail", "Pass"],
              filled=True, rounded=True, fontsize=8, ax=ax, impurity=False,
              proportion=False)
    ax.set_title("Decision Tree Structure (max_depth=4)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: plots/{filename}")


def plot_confusion(name, model, X, y, filename):
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)
    _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    feature_names = X_train.columns.tolist()
    X_train_s, X_val_s, X_test_s = scale(X_train, X_val, X_test)

    lr_l2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_l1 = LogisticRegression(C=0.1, l1_ratio=1, solver="saga", max_iter=2000, random_state=42)
    dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_full = DecisionTreeClassifier(max_depth=None, random_state=42)

    named_models = [
        ("LR (L2, C=1.0)", lr_l2),
        ("LR (L1, C=0.1)", lr_l1),
        ("Decision Tree (depth=4)", dt_pruned),
        ("Decision Tree (full)", dt_full),
    ]

    print("=== 5-Fold Cross-Validation (Train Set) ===")
    cv_df = cross_validate_models(named_models, X_train_s, y_train)
    print(cv_df.to_string())
    plot_cv_comparison(cv_df)

    for _, model in named_models:
        model.fit(X_train_s, y_train)

    print("\n=== Validation Set Results ===")
    val_results = [evaluate(name, model, X_val_s, y_val) for name, model in named_models]
    val_df = pd.DataFrame(val_results).set_index("Model")
    print(val_df.to_string())

    print("\n=== Test Set Results ===")
    test_results = [evaluate(name, model, X_test_s, y_test) for name, model in named_models]
    test_df = pd.DataFrame(test_results).set_index("Model")
    print(test_df.to_string())

    plot_metric_comparison(val_df, test_df)

    plot_lr_feature_importance(lr_l2, feature_names, "LR (L2, C=1.0)", "fi_lr_l2.png")
    plot_lr_feature_importance(lr_l1, feature_names, "LR (L1, C=0.1)", "fi_lr_l1.png")
    plot_dt_feature_importance(dt_pruned, feature_names, "Decision Tree (depth=4)", "fi_dt_pruned.png")
    plot_dt_feature_importance(dt_full, feature_names, "Decision Tree (full)", "fi_dt_full.png")

    plot_decision_tree_structure(dt_pruned, feature_names, "decision_tree_structure.png")

    for name, model in named_models:
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "").replace(".", "")
        plot_confusion(name, model, X_test_s, y_test, f"cm_{safe}.png")

    print("\nAll Week 7 plots saved to plots/")


if __name__ == "__main__":
    main()
