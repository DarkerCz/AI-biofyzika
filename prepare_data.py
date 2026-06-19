"""
Stáhne a připraví všech 6 datasetů pro seminář.
Spusťte jednou před spuštěním aplikace:
    python prepare_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import yaml
import warnings
warnings.filterwarnings("ignore")


def load_settings():
    with open("settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def augment_dataset(df, target_col, target_n=3000, seed=42):
    """Augment dataset to target_n rows. Continuous columns get 5% Gaussian noise."""
    if len(df) >= target_n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    continuous = [c for c in df.columns if c != target_col and df[c].nunique() > 10]
    augmented = [df]
    while sum(len(a) for a in augmented) < target_n:
        sample = df.sample(n=min(len(df), target_n), replace=True,
                           random_state=seed + len(augmented))
        sample = sample.copy()
        for col in continuous:
            std = df[col].std() * 0.05
            if std > 0:
                sample[col] = (sample[col] + rng.normal(0, std, len(sample))).round(4)
        augmented.append(sample)
    result = pd.concat(augmented, ignore_index=True)
    return result.sample(n=target_n, random_state=seed).reset_index(drop=True)


def shuffle_labels(df, target_col, seed=42):
    df2 = df.copy()
    df2[target_col] = df2[target_col].sample(frac=1, random_state=seed).values
    return df2


def save_group_data(group_id, df, target_col, correct_dataset, seed=42):
    path = Path(f"data/group{group_id}")
    path.mkdir(parents=True, exist_ok=True)

    try:
        train_df, test_df = train_test_split(
            df, test_size=0.25, random_state=seed, stratify=df[target_col]
        )
    except ValueError:
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=seed)

    correct_df = train_df.reset_index(drop=True)
    corrupted_df = shuffle_labels(train_df, target_col, seed=seed).reset_index(drop=True)

    if correct_dataset == "A":
        dataset_a, dataset_b = correct_df, corrupted_df
    else:
        dataset_a, dataset_b = corrupted_df, correct_df

    dataset_a.to_csv(path / "dataset_A.csv", index=False)
    dataset_b.to_csv(path / "dataset_B.csv", index=False)
    test_df.reset_index(drop=True).to_csv(path / "test.csv", index=False)

    pos_rate = df[target_col].mean()
    print(f"  Skupina {group_id}: {len(train_df)} trénovací, "
          f"{len(test_df)} testovací, {pos_rate:.1%} pozitivních")


# ── Group 1: Heart Disease ────────────────────────────────────────────────────

def prepare_group1():
    print("Skupina 1: Srdeční choroba...")
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = cols
    except Exception:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        df = pd.read_csv(url, header=None, names=cols, na_values="?")
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)
    return augment_dataset(df, "target"), "target"


# ── Group 2: Diabetes ─────────────────────────────────────────────────────────

def prepare_group2():
    print("Skupina 2: Diabetes...")
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=34)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = cols
    except Exception:
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
               "pima-indians-diabetes.data.csv")
        df = pd.read_csv(url, header=None, names=cols)
    return augment_dataset(df.reset_index(drop=True), "Outcome"), "Outcome"


# ── Group 3: Breast Cancer ────────────────────────────────────────────────────

def prepare_group3():
    print("Skupina 3: Rakovina prsu...")
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["diagnosis"] = (1 - bc.target)   # 0 = benigní, 1 = maligní
    return augment_dataset(df, "diagnosis"), "diagnosis"


# ── Group 4: Heart Failure ────────────────────────────────────────────────────

def create_synthetic_heart_failure():
    rng = np.random.default_rng(456)
    n = 299
    age = rng.uniform(40, 95, n)
    ef = rng.uniform(14, 80, n)
    creatinine = rng.uniform(0.5, 9.4, n)
    sodium = rng.uniform(113, 148, n)
    time = rng.integers(4, 285, n).astype(float)

    logit = (-4
             + 0.04 * (age - 60)
             - 0.05 * (ef - 38)
             + 0.3 * creatinine
             - 0.02 * (sodium - 136)
             - 0.01 * time
             + rng.normal(0, 0.4, n))
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "age": age.round(0),
        "anaemia": rng.integers(0, 2, n).astype(float),
        "creatinine_phosphokinase": rng.integers(23, 7861, n).astype(float),
        "diabetes": rng.integers(0, 2, n).astype(float),
        "ejection_fraction": ef.round(0),
        "high_blood_pressure": rng.integers(0, 2, n).astype(float),
        "platelets": rng.uniform(25100, 850000, n).round(0),
        "serum_creatinine": creatinine.round(1),
        "serum_sodium": sodium.round(0),
        "sex": rng.integers(0, 2, n).astype(float),
        "smoking": rng.integers(0, 2, n).astype(float),
        "time": time,
        "DEATH_EVENT": target,
    })


def prepare_group4():
    print("Skupina 4: Srdeční selhání...")
    df = None
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=519)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        # ucimlrepo může vrátit jiný název sloupce — normalizujeme
        target_name = ds.data.targets.columns[0]
        if target_name != "DEATH_EVENT":
            df = df.rename(columns={target_name: "DEATH_EVENT"})
    except Exception as e:
        print(f"  ucimlrepo selhalo ({e}), zkouším přímé stažení...")

    if df is None or "DEATH_EVENT" not in df.columns:
        try:
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "00519/heart_failure_clinical_records_dataset.csv")
            df = pd.read_csv(url)
        except Exception as e:
            print(f"  URL selhalo ({e}), používám syntetická data...")
            df = create_synthetic_heart_failure()

    return augment_dataset(df.dropna().reset_index(drop=True), "DEATH_EVENT"), "DEATH_EVENT"


# ── Group 5: Readmission (syntetická data) ────────────────────────────────────

def prepare_group5():
    print("Skupina 5: Readmise do nemocnice (syntetická data)...")
    rng = np.random.default_rng(42)
    n = 3000

    age = rng.integers(20, 90, n).astype(float)
    time_in_hospital = rng.integers(1, 15, n).astype(float)
    n_emergency = rng.integers(0, 6, n).astype(float)
    n_meds = rng.integers(1, 30, n).astype(float)
    HbA1c = rng.uniform(4.5, 14.0, n)
    glucose = rng.uniform(70, 400, n)
    BMI = rng.uniform(16.0, 52.0, n)
    n_diagnoses = rng.integers(1, 9, n).astype(float)
    creatinine = rng.uniform(0.5, 4.5, n)

    logit = (-4.5
             + 0.6 * n_emergency
             + 0.15 * time_in_hospital
             + 0.05 * (HbA1c - 7)
             + 0.008 * (glucose - 140)
             + 0.02 * (age - 60)
             + rng.normal(0, 0.5, n))
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "vek": age,
        "delka_hospitalizace": time_in_hospital,
        "pocet_urgentni_prijmu": n_emergency,
        "pocet_leku": n_meds,
        "HbA1c": HbA1c.round(1),
        "glukoza": glucose.round(1),
        "BMI": BMI.round(1),
        "pocet_diagnoz": n_diagnoses,
        "kreatinin": creatinine.round(2),
        "readmitted_30": target,
    })
    return df, "readmitted_30"


# ── Group 6: Sepsis (syntetická data) ─────────────────────────────────────────

def prepare_group6():
    print("Skupina 6: Sepse (syntetická data)...")
    rng = np.random.default_rng(123)
    n = 3000

    HR = rng.uniform(50, 160, n)
    O2Sat = rng.uniform(85, 100, n)
    Temp = rng.uniform(35.5, 41.0, n)
    MAP = rng.uniform(40, 120, n)
    Resp = rng.uniform(10, 40, n)
    Lactate = rng.uniform(0.5, 10.0, n)
    WBC = rng.uniform(2, 30, n)
    Creatinine = rng.uniform(0.5, 6.0, n)
    Bilirubin = rng.uniform(0.2, 8.0, n)
    SOFA = rng.integers(0, 16, n).astype(float)
    age = rng.integers(18, 90, n).astype(float)

    logit = (-3.0
             + 0.35 * SOFA
             + 0.25 * Lactate
             - 0.025 * MAP
             + 0.12 * Creatinine
             + 0.02 * (age - 60)
             + rng.normal(0, 0.4, n))
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "SF": HR.round(1),
        "SpO2": O2Sat.round(1),
        "Temp": Temp.round(1),
        "MAP": MAP.round(1),
        "Resp": Resp.round(1),
        "Laktat": Lactate.round(2),
        "WBC": WBC.round(1),
        "Kreatinin": Creatinine.round(2),
        "Bilirubin": Bilirubin.round(2),
        "SOFA": SOFA,
        "vek": age,
        "SepsisLabel": target,
    })
    return df, "SepsisLabel"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings()
    Path("data").mkdir(exist_ok=True)

    preparers = [
        (1, prepare_group1),
        (2, prepare_group2),
        (3, prepare_group3),
        (4, prepare_group4),
        (5, prepare_group5),
        (6, prepare_group6),
    ]

    for group_id, func in preparers:
        group = settings["groups"][group_id]
        df, target_col = func()
        save_group_data(
            group_id=group_id,
            df=df,
            target_col=target_col,
            correct_dataset=group["correct_dataset"],
        )

    print("\nHotovo! Spusťte aplikaci: streamlit run app.py")


if __name__ == "__main__":
    main()
