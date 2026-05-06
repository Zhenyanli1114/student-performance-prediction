import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")

DATA_PATH = "data/student-mat.csv"
PLOTS_DIR = "plots"


def load_raw():
    return pd.read_csv(DATA_PATH, sep=";")


def plot_class_distribution(df):
    labels = ["Fail (G3 < 10)", "Pass (G3 >= 10)"]
    counts = [(df["G3"] < 10).sum(), (df["G3"] >= 10).sum()]

    _, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=["#d9534f", "#5cb85c"], edgecolor="white", width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(count), ha="center", va="bottom", fontsize=11)
    ax.set_title("Class Distribution (Pass vs Fail)", fontsize=13)
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, max(counts) + 30)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("Saved: plots/class_distribution.png")


def plot_feature_distributions(df):
    features = ["studytime", "absences", "failures"]
    titles = ["Study Time (1-4 scale)", "Number of Absences", "Number of Past Failures"]

    _, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, feat, title in zip(axes, features, titles):
        ax.hist(df[feat], bins=15, color="#4a90d9", edgecolor="white", rwidth=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Distribution of Key Features", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/feature_distributions.png")


def plot_correlation_heatmap(df):
    numeric_cols = ["studytime", "absences", "failures", "age", "Medu", "Fedu",
                    "traveltime", "freetime", "goout", "Dalc", "Walc", "health"]
    target = (df["G3"] >= 10).astype(int).rename("pass")
    corr_df = df[numeric_cols].copy()
    corr_df["pass"] = target
    corr = corr_df.corr()

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap (Numeric Features + Target)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved: plots/correlation_heatmap.png")


def plot_boxplots_by_outcome(df):
    df = df.copy()
    df["outcome"] = df["G3"].apply(lambda g: "Pass" if g >= 10 else "Fail")

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    palette = {"Pass": "#5cb85c", "Fail": "#d9534f"}

    sns.boxplot(data=df, x="outcome", y="absences", hue="outcome",
                palette=palette, legend=False, ax=axes[0])
    axes[0].set_title("Absences by Outcome")
    axes[0].set_xlabel("")
    axes[0].spines[["top", "right"]].set_visible(False)

    sns.boxplot(data=df, x="outcome", y="studytime", hue="outcome",
                palette=palette, legend=False, ax=axes[1])
    axes[1].set_title("Study Time by Outcome")
    axes[1].set_xlabel("")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Key Features Grouped by Pass/Fail", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "boxplots_by_outcome.png"), dpi=150)
    plt.close()
    print("Saved: plots/boxplots_by_outcome.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_raw()
    print(f"Raw dataset shape: {df.shape}")

    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_outcome(df)

    print("\nAll EDA plots saved to plots/")


if __name__ == "__main__":
    main()
