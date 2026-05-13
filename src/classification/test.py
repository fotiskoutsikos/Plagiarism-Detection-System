import importlib
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, fbeta_score
# Resolve repository root & logging
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

sys.path.insert(0, str(repo_root / "src"))


from utils.categorization import fbeta_score_curve
from utils.constants import RANDOM_STATE, NUM_K_FOLDS, BETA

df = pd.read_parquet("results/classification/classifier_features.parquet")

# Feature groups
dist_cols = [c for c in df.columns if "distance" in c]
delta_cols = [c for c in df.columns if any(x in c for x in ["delta", "stable", "volatile", "active"])]
vocal_cols = [
    c for c in df.columns
    if c in {"pair_vocal_valid", "vocal_ratio_ori", "vocal_ratio_mod", "vocal_valid_ori", "vocal_valid_mod"}
]

feature_sets = {
    "1. Distances Only": dist_cols,
    "2. Vocal Only": vocal_cols,
    "3. Distances + Vocal": dist_cols + vocal_cols,
    "4. Deltas + Vocal": delta_cols + vocal_cols,
    "5. All Features": dist_cols + delta_cols + vocal_cols,
}

y = df["is_plagiarised"].astype(int).values
groups = df["filename_ori"].values

def find_best_threshold(y_true, y_prob, beta=BETA):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = fbeta_score_curve(precision, recall, beta)
    if len(scores) <= 1:
        return 0.5
    return float(thresholds[int(np.argmax(scores[:-1]))])

print("=" * 70)
print("FEATURE ABLATION (MLP CLASSIFIER) — VOCAL CHECK")
print("=" * 70)

for name, cols in feature_sets.items():
    X = df[cols].values.astype(np.float64)
    sgkf = StratifiedGroupKFold(
        n_splits=NUM_K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores = []
    for tr, te in sgkf.split(X, y, groups):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate="adaptive",
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=RANDOM_STATE,
                verbose=False,
            )),
        ])

        pipe.fit(X_train, y_train)

        prob_train = pipe.predict_proba(X_train)[:, 1]
        prob_test  = pipe.predict_proba(X_test)[:, 1]

        thr  = find_best_threshold(y_train, prob_train)
        pred = (prob_test >= thr).astype(int)

        fold_scores.append(fbeta_score(y_test, pred, beta=BETA, zero_division=0))

    print(f"{name:<28} | Mean F0.5 = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")