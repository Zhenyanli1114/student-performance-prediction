import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import seaborn as sns

matplotlib.use("Agg")

DATA_DIR = "data"
PLOTS_DIR = "plots"


def load_splits():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train, y_train = train.drop(columns=["pass"]), train["pass"]
    X_val, y_val = val.drop(columns=["pass"]), val["pass"]
    X_test, y_test = test.drop(columns=["pass"]), test["pass"]
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


def plot_confusion(name, model, X, y, filename):
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
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
    X_train_s, X_val_s, X_test_s = scale(X_train, X_val, X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)

    print("=== Validation Set Results ===")
    val_results = [
        evaluate("Logistic Regression", lr, X_val_s, y_val),
        evaluate("KNN (k=5)", knn, X_val_s, y_val),
    ]
    val_df = pd.DataFrame(val_results).set_index("Model")
    print(val_df.to_string())

    plot_confusion("Logistic Regression", lr, X_val_s, y_val, "cm_logistic_regression.png")
    plot_confusion("KNN (k=5)", knn, X_val_s, y_val, "cm_knn.png")

    print("\n=== Test Set Results ===")
    test_results = [
        evaluate("Logistic Regression", lr, X_test_s, y_test),
        evaluate("KNN (k=5)", knn, X_test_s, y_test),
    ]
    test_df = pd.DataFrame(test_results).set_index("Model")
    print(test_df.to_string())


if __name__ == "__main__":
    main()
