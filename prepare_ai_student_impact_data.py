from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_DIR = Path(__file__).resolve().parent / "data" / "ai_student_impact"
RAW_FILE = DATASET_DIR / "ai_student_impact_dataset (1).csv"
TARGET_COLUMN = "Burnout_Risk_Level"
DROP_COLUMNS = ["Student_ID"]
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def to_dataframe(transformed, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(transformed, columns=columns)


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_df = df.drop(columns=[TARGET_COLUMN, *[col for col in DROP_COLUMNS if col in df.columns]])
    target = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        feature_df,
        target,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor(feature_df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_processed_df = to_dataframe(X_train_processed, feature_names)
    X_val_processed_df = to_dataframe(X_val_processed, feature_names)
    y_train_df = y_train.reset_index(drop=True).to_frame(name=TARGET_COLUMN)
    y_val_df = y_val.reset_index(drop=True).to_frame(name=TARGET_COLUMN)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    X_train_processed_df.to_csv(DATASET_DIR / "X_train_processed.csv", index=False)
    y_train_df.to_csv(DATASET_DIR / "y_train_processed.csv", index=False)
    X_val_processed_df.to_csv(DATASET_DIR / "X_val_processed.csv", index=False)
    y_val_df.to_csv(DATASET_DIR / "y_val_processed.csv", index=False)

    print("Prepared ai_student_impact dataset.")
    print(f"Train rows: {len(X_train_processed_df)}")
    print(f"Validation rows: {len(X_val_processed_df)}")
    print(f"Processed feature count: {len(feature_names)}")
    print(f"Saved files under: {DATASET_DIR}")


if __name__ == "__main__":
    main()
