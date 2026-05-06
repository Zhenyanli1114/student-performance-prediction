import os
import io
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "student-mat.csv")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

BINARY_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]

MULTI_COLS = ["Mjob", "Fjob", "reason", "guardian"]


def download_data():
    if os.path.exists(RAW_PATH):
        return
    print("Downloading dataset...")
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("student-mat.csv") as f:
            content = f.read().decode("utf-8")
    with open(RAW_PATH, "w") as f:
        f.write(content)
    print(f"Saved to {RAW_PATH}")


def load_and_clean():
    df = pd.read_csv(RAW_PATH, sep=";")

    # target: pass if final grade >= 10
    df["pass"] = (df["G3"] >= 10).astype(int)
    df = df.drop(columns=["G1", "G2", "G3"])

    # encode yes/no and other binary categoricals
    binary_map = {"yes": 1, "no": 0, "GP": 1, "MS": 0, "F": 1, "M": 0,
                  "U": 1, "R": 0, "GT3": 1, "LE3": 0, "T": 1, "A": 0}
    for col in BINARY_COLS:
        df[col] = df[col].map(binary_map).fillna(df[col])

    # one-hot encode multi-class categoricals
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True)

    return df


def split_and_save(df):
    X = df.drop(columns=["pass"])
    y = df["pass"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    for split, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out = X_split.copy()
        out["pass"] = y_split.values
        out.to_csv(os.path.join(DATA_DIR, f"{split}.csv"), index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_data()
    df = load_and_clean()

    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Class balance:\n{df['pass'].value_counts().rename({0: 'fail', 1: 'pass'})}")

    splits = split_and_save(df)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    print(f"\nSplit sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print("Splits saved to data/train.csv, data/val.csv, data/test.csv")


if __name__ == "__main__":
    main()
