import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def preprocess(train_path, test_path, target_column, display_columns=None):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    le = LabelEncoder()
    if y_train.dtype == object or y_test.dtype == object:
        le.fit(pd.concat([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
    else:
        y_train_enc = y_train.values
        y_test_enc = y_test.values

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).select_dtypes(include=[np.number])
    X_test = X_test.fillna(medians)[X_train.columns]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if display_columns:
        avail = [c for c in display_columns if c in test_df.columns and c != target_column]
        display_df = test_df[avail].reset_index(drop=True)
    else:
        display_df = X_test.reset_index(drop=True)

    return X_train_sc, y_train_enc, X_test_sc, y_test_enc, display_df


def build_mlp(mlp_cfg):
    return MLPClassifier(
        hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [128, 64, 32])),
        max_iter=1,
        warm_start=True,
        random_state=mlp_cfg.get("random_state", 42),
    )


def predict(mlp, X_test_sc, y_test_enc, display_df):
    y_pred = mlp.predict(X_test_sc)
    y_prob = mlp.predict_proba(X_test_sc)[:, 1]
    cm = confusion_matrix(y_test_enc, y_pred, labels=[0, 1]).tolist()

    table_df = display_df.copy()
    table_df["__pred__"] = y_pred
    table_df["__prob__"] = (y_prob * 100).round(1)
    table_df["__true__"] = y_test_enc

    return {
        "accuracy": float(accuracy_score(y_test_enc, y_pred)),
        "confusion_matrix": cm,
        "n_test": len(y_test_enc),
        "table": table_df.to_dict("records"),
    }
