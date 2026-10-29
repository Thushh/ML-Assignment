"""Step 2 - split, hyperparameter tuning, model training; saves the models.

Run:  python -m src.train        (after src.preprocess)

Split by time (never at random, which would let the model learn from the future):
    train       Feb-2023 .. Mar-2024   fit candidate models
    validation  Apr-2024 .. Jun-2024   choose hyperparameters (macro-F1)
    test        Jul-2024 .. Dec-2024   untouched here; used only by src/evaluate.py
After tuning, each model is refitted on train + validation with its best settings and saved.

Main model: NGBoost (Natural Gradient Boosting). It predicts a full probability distribution
of next month's capacity factor, Normal(mean, sd), and the tier probabilities are read off
that distribution. Compared with:
    two other algorithms not taught in the module: Gaussian Process and Naive Bayes
    three models covered in the module:             EBM, Random Forest and XGBoost
    plus the "same as last month" rule (in evaluate.py)
"""
import itertools
import json
import pickle
import platform
import time

import interpret
import ngboost
import pandas as pd
import sklearn
import xgboost
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocess import (FEATURES, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, TARGET, NGBoostTiers, load_split,
                         make_preprocessor)


# Each model: a base estimator and a small grid searched on the validation months.
MODELS = {
    "NGBoost": (
        NGBoostTiers(learning_rate=0.03),
        {"max_depth": [3, 5],               # depth of each small regression tree (the weak learner)
         "n_estimators": [300, 800]},       # boosting rounds
    ),
    "Gaussian Process": (
        # kernel hyperparameters (length scale, amplitude) are fitted by maximising the marginal likelihood
        Pipeline([("prep", make_preprocessor()),
                  ("model", GaussianProcessClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
        {"model__kernel": [ConstantKernel(1.0) * RBF(5.0), ConstantKernel(1.0) * Matern(5.0, nu=1.5)]},
    ),
    "Naive Bayes": (
        Pipeline([("prep", make_preprocessor()), ("model", GaussianNB())]),
        {"model__var_smoothing": [1e-9, 1e-3, 1e-1]},   # variance added to every feature for stability
    ),
    "EBM": (
        ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"interactions": [0, 10], "learning_rate": [0.01, 0.04]},
    ),
    "Random Forest": (
        Pipeline([("prep", make_preprocessor()),
                  ("model", RandomForestClassifier(n_estimators=300, max_features=0.5, min_samples_leaf=3,
                                                   n_jobs=-1, random_state=RANDOM_STATE))]),
        {"model__max_depth": [8, 14, None]},
    ),
    "XGBoost": (
        Pipeline([("prep", make_preprocessor()),
                  ("model", XGBClassifier(n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                          reg_lambda=2.0, objective="multi:softprob", tree_method="hist",
                                          n_jobs=-1, random_state=RANDOM_STATE))]),
        {"model__max_depth": [3, 5]},
    ),
}
FILE_NAMES = {"NGBoost": "ngboost.pkl", "Gaussian Process": "gaussian_process.pkl", "Naive Bayes": "naive_bayes.pkl",
              "EBM": "ebm.pkl", "Random Forest": "random_forest.pkl", "XGBoost": "xgboost.pkl"}

# A Gaussian Process costs O(n^3) in the number of training rows, so it is trained on a random
# sample (a standard way to keep GPs practical); every other model uses all rows.
SAMPLE_ROWS = {"Gaussian Process": 2000}


def rows_for(name, data):
    n = SAMPLE_ROWS.get(name)
    return data.sample(n, random_state=RANDOM_STATE) if n and len(data) > n else data


def target_for(name, data):
    """NGBoost learns the capacity factor itself (a number); the others learn the tier directly."""
    return data["cf"] if name == "NGBoost" else data[TARGET]


def tune(name, base, grid, train, val):
    """Try every combination on the validation months; return the best settings and all scores."""
    rows = []
    train = rows_for(name, train)
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = clone(base).set_params(**params).fit(train[FEATURES], target_for(name, train))
        score = f1_score(val[TARGET], model.predict(val[FEATURES]), average="macro")
        rows.append({"model": name, **{k.replace("model__", ""): str(v) for k, v in params.items()}, "val_macro_f1": round(score, 4)})
        print(f"  {name:20s} {params} -> validation macro-F1 {score:.4f}")
    best = max(range(len(rows)), key=lambda i: rows[i]["val_macro_f1"])
    return dict(zip(grid.keys(), list(itertools.product(*grid.values()))[best])), rows


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    train, val, test = load_split()
    print(f"rows  train={len(train)}  validation={len(val)}  test={len(test)}")
    train_val = pd.concat([train, val])

    best_params, all_rows, fit_seconds = {}, [], {}
    for name, (base, grid) in MODELS.items():
        print(f"tuning {name} ...")
        params, rows = tune(name, base, grid, train, val)
        all_rows += rows
        best_params[name] = {k.replace("model__", ""): v if isinstance(v, (int, float, type(None))) else str(v)
                             for k, v in params.items()}
        t0 = time.time()
        fit_data = rows_for(name, train_val)                                    # refit on train + validation
        final = clone(base).set_params(**params).fit(fit_data[FEATURES], target_for(name, fit_data))
        fit_seconds[name] = round(time.time() - t0, 1)
        with open(MODELS_DIR / FILE_NAMES[name], "wb") as fh:
            pickle.dump(final, fh)

    pd.DataFrame(all_rows).to_csv(OUTPUTS_DIR / "validation_scores.csv", index=False)
    meta = {
        "task": "3-class classification of next-month capacity-factor tier (Low / Moderate / High)",
        "main_model": "NGBoost", "model_files": FILE_NAMES, "features": FEATURES,
        "split": {"train": "Feb-2023 .. Mar-2024", "validation": "Apr-2024 .. Jun-2024", "test": "Jul-2024 .. Dec-2024",
                  "final_models_trained_on": "train + validation (Feb-2023 .. Jun-2024)",
                  "rows": {"train": len(train), "validation": len(val), "test": len(test)}},
        "best_params": best_params, "fit_seconds": fit_seconds, "training_row_samples": SAMPLE_ROWS,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "ngboost": ngboost.__version__,
                     "xgboost": xgboost.__version__, "interpret": interpret.__version__, "pandas": pd.__version__},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(best_params, indent=2))
    print(f"models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
