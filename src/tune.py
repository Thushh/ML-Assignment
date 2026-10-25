"""FINAL ASSIGNMENT - hyperparameter tuning for CatBoost and LightGBM with time-series CV.

Run:  python -m src.tune            (about 10-15 minutes; CatBoost is the slow part)
Outputs:
    models/final/best_params.json        the chosen settings (read by src/final_train.py)
    reports/final/tuning_results.csv     every configuration tried, with CV mean and std

Method
------
* Starting point = the hand-designed values in src/final_models.py (reasoning written there).
* Random search around them: each candidate is scored by mean macro-F1 over EXPANDING-WINDOW
  time folds (train on earlier months, validate on the next two). No test-period data is used.
* The search also decides whether each model should use the 324-level `plant_id` categorical.
* Tuning uses ONLY folds 1-3 (validation Jul-Dec 2023, TUNE_CV_STARTS). Folds 4-6 are reserved for
  model selection in src/final_train.py, so the selection score is not inflated by tuning.
* LightGBM: 30 random candidates; CatBoost (~20-30 s per fit): 7 random candidates.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

from .config import FINAL_INPUT, FINAL_MODELS_DIR, FINAL_REPORTS_DIR, RANDOM_STATE, TARGET, TUNE_CV_STARTS
from .final_models import (CATBOOST_PARAMS, LIGHTGBM_PARAMS, catboost_pipeline, lightgbm_pipeline,
                           time_folds)
from .train import load_split

LIGHTGBM_SPACE = {
    "n_estimators": [300, 600, 900],
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [7, 15, 31],
    "max_depth": [3, 4, 5, 6],
    "min_child_samples": [20, 30, 50, 80],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.5, 2.0, 5.0],
    "use_plant_id": [False, True],
}
CATBOOST_SPACE = {
    "iterations": [500, 800],
    "learning_rate": [0.03, 0.05],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [3.0, 5.0, 10.0],
    "use_plant_id": [True, False],
}


def cv_score(pipe, train, folds):
    scores = []
    for tr, va in folds:
        m = clone(pipe).fit(train.iloc[tr][FINAL_INPUT], train.iloc[tr][TARGET])
        scores.append(f1_score(train.iloc[va][TARGET], m.predict(train.iloc[va][FINAL_INPUT]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def search(name, make, designed, space, n_iter, folds, train, rows):
    tunable = {k: designed[k] for k in space}          # the hand-designed point is always tried first
    candidates = [tunable] + list(ParameterSampler(space, n_iter=n_iter, random_state=RANDOM_STATE))
    best = None
    for i, params in enumerate(candidates):
        t0 = time.time()
        mean, std = cv_score(make(params), train, folds)
        rows.append({"model": name, "candidate": "designed" if i == 0 else f"random {i}",
                     "cv_macro_f1": round(mean, 4), "cv_std": round(std, 4),
                     "seconds": round(time.time() - t0, 1), **params})
        print(f"{name:9s} {i:2d}/{len(candidates) - 1}  CV macro-F1 {mean:.4f} ± {std:.4f}  {params}")
        if best is None or mean > best[0]:
            best = (mean, params)
    return best


def plant_id_ablation(best, train, folds, rows):
    """Re-score each model's best settings with `use_plant_id` flipped, on the same folds,
    so the decision to use (or not use) plant identity rests on a direct comparison."""
    out = {}
    for name, make, f in (("LightGBM", lightgbm_pipeline, folds), ("CatBoost", catboost_pipeline, folds)):
        params = dict(best[name])
        base, _ = cv_score(make(params), train, f)
        flipped = dict(params, use_plant_id=not params["use_plant_id"])
        alt, alt_std = cv_score(make(flipped), train, f)
        rows.append({"model": name, "candidate": "ablation: plant_id flipped", "cv_macro_f1": round(alt, 4),
                     "cv_std": round(alt_std, 4), "seconds": None, **flipped})
        out[name] = {"with_plant_id" if params["use_plant_id"] else "without_plant_id": round(base, 4),
                     "without_plant_id" if params["use_plant_id"] else "with_plant_id": round(alt, 4)}
        print(f"{name} plant_id ablation: {out[name]}")
    return out


def ablation_only():
    """Run just the ablation on the saved best params (avoids repeating the full search)."""
    _, train, _ = load_split()
    folds = time_folds(train, TUNE_CV_STARTS)
    best = json.loads((FINAL_MODELS_DIR / "best_params.json").read_text())
    rows = []
    best["_plant_id_ablation"] = plant_id_ablation(best, train, folds, rows)
    (FINAL_MODELS_DIR / "best_params.json").write_text(json.dumps(best, indent=2, default=str))
    old = pd.read_csv(FINAL_REPORTS_DIR / "tuning_results.csv")
    old = old[old.candidate != "ablation: plant_id flipped"]
    pd.concat([old, pd.DataFrame(rows)]).to_csv(FINAL_REPORTS_DIR / "tuning_results.csv", index=False)


def main():
    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _, train, _ = load_split()
    folds = time_folds(train, TUNE_CV_STARTS)
    rows = []

    lgb_best = search("LightGBM", lightgbm_pipeline, LIGHTGBM_PARAMS, LIGHTGBM_SPACE, 30, folds, train, rows)
    cb_best = search("CatBoost", catboost_pipeline, CATBOOST_PARAMS, CATBOOST_SPACE, 7, folds, train, rows)

    best = {"LightGBM": lgb_best[1], "CatBoost": cb_best[1],
            "_cv": {"LightGBM": round(lgb_best[0], 4), "CatBoost": round(cb_best[0], 4),
                    "note": "scored on tuning folds 1-3 only (validation Jul-Dec 2023)"}}
    best["_plant_id_ablation"] = plant_id_ablation(best, train, folds, rows)
    (FINAL_MODELS_DIR / "best_params.json").write_text(json.dumps(best, indent=2, default=str))
    pd.DataFrame(rows).to_csv(FINAL_REPORTS_DIR / "tuning_results.csv", index=False)
    print("\nbest:", json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    import sys
    ablation_only() if "--ablation-only" in sys.argv else main()
