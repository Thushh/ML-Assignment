"""CLASS ACTIVITY - SHAP (global + local) and LIME (local) for the saved soft-voting pipeline.

Run:  python -m src.explain_ensemble        (after src.train has saved models/model.pkl)
(The final assignment's CatBoost/LightGBM SHAP lives in src/explain.py.)

Both methods explain the WHOLE pipeline (imputation, scaling, one-hot, RF + XGBoost, averaging)
on the 13 original features, in probability units - exactly the model the web app serves.
SHAP's tree explainer is not used because shap 0.49 cannot read XGBoost 3.x models (per-class
base_score) and the fix needs numpy 2, which the pinned stack does not support.

Outputs:
    reports/figures/shap_global_importance.png  mean |SHAP| per feature and class (test months)
    reports/figures/shap_beeswarm_low.png       how each feature pushes towards / away from "Low"
    reports/figures/shap_waterfall_<case>.png   one prediction broken into feature contributions
    reports/figures/lime_<case>.png             LIME local surrogate for the same prediction
    reports/explanations.json                   all numbers behind the figures
"""
import json
import pickle

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from .config import (CATEGORICAL_FEATURES, CATEGORIES, CLASS_COLORS, CLASS_NAMES, FEATURES,
                     MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, RANDOM_STATE, SEASON_BY_MONTH, TARGET)
from .features import cf_to_class
from .plotting import plt, save
from .train import load_split

SEASONS = list(dict.fromkeys(SEASON_BY_MONTH.values()))
LEVELS = {"category": CATEGORIES, "season": SEASONS}
CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]
N_BACKGROUND = 100   # training rows that stand in for "feature unknown" when SHAP masks a feature
N_GLOBAL = 200       # random test rows explained for the global picture (~0.8 s each)


# --- SHAP and LIME both need a purely numeric matrix: categories <-> integer codes -------------
def to_numeric(X: pd.DataFrame) -> np.ndarray:
    X = X[FEATURES].copy()
    for c, levels in LEVELS.items():
        X[c] = X[c].map({v: i for i, v in enumerate(levels)})
    return X.to_numpy(dtype=float)


def to_frame(A: np.ndarray) -> pd.DataFrame:
    X = pd.DataFrame(A, columns=FEATURES)
    for c, levels in LEVELS.items():
        X[c] = [levels[int(round(v))] for v in X[c]]
    return X


def proba_fn(model):
    return lambda A: model.predict_proba(to_frame(np.atleast_2d(A)))


def display_values(row: pd.Series) -> list:
    return [v if isinstance(v, str) else ("missing" if pd.isna(v) else round(float(v), 3)) for v in row]


# --- SHAP --------------------------------------------------------------------------------------
def shap_explainer(model, train):
    bg = to_numeric(train[FEATURES].sample(N_BACKGROUND, random_state=RANDOM_STATE))
    masker = shap.maskers.Independent(bg, max_samples=N_BACKGROUND)
    return shap.PermutationExplainer(proba_fn(model), masker, feature_names=FEATURES,
                                     seed=RANDOM_STATE)


def global_shap(explainer, test):
    sample = test.sample(min(N_GLOBAL, len(test)), random_state=RANDOM_STATE)
    sv = explainer(to_numeric(sample), max_evals=2 * len(FEATURES) * 5 + 1)   # 5 permutations/row
    mean_abs = pd.DataFrame(np.abs(sv.values).mean(0), index=FEATURES, columns=CLASS_NAMES)
    mean_abs = mean_abs.loc[mean_abs.sum(1).sort_values().index]

    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(mean_abs))
    for k, cls in enumerate(CLASS_NAMES):
        ax.barh(mean_abs.index, mean_abs[cls], left=left, color=CLASS_COLORS[k], label=cls, height=0.6,
                edgecolor="white", linewidth=1)
        left += mean_abs[cls].values
    ax.set_xlabel("Mean |SHAP value| (change in class probability)")
    ax.set_title(f"SHAP global importance, soft-voting ensemble ({len(sample)} test plant-months)")
    ax.legend(ncol=3, loc="lower right")
    save(fig, "shap_global_importance.png")

    # beeswarm for the class planners care about most: will the plant under-deliver?
    low = shap.Explanation(sv.values[:, :, 0], base_values=sv.base_values[:, 0],
                           data=to_numeric(sample), feature_names=FEATURES)
    low.data[:, CAT_IDX] = np.nan        # category codes have no order -> drawn grey
    plt.figure(figsize=(9, 5.5))
    shap.plots.beeswarm(low, max_display=13, show=False, plot_size=None)
    plt.title('SHAP beeswarm: what pushes a plant-month towards "Low"', loc="left")
    save(plt.gcf(), "shap_beeswarm_low.png")
    return mean_abs.iloc[::-1].round(4).to_dict("index")


def local_shap(explainer, row: pd.DataFrame, cls: int, case: str):
    sv = explainer(to_numeric(row), max_evals=2000)   # 13 features: enough for ~75 full permutations
    one = shap.Explanation(sv.values[0, :, cls], base_values=sv.base_values[0, cls],
                           data=to_numeric(row)[0], display_data=np.array(display_values(row.iloc[0]), dtype=object),
                           feature_names=FEATURES)
    plt.figure(figsize=(9, 5.5))
    shap.plots.waterfall(one, max_display=13, show=False)
    plt.title(f"SHAP: why P({CLASS_NAMES[cls]}) = {one.base_values + one.values.sum():.2f} for {case}",
              loc="left", fontsize=11)
    save(plt.gcf(), f"shap_waterfall_{slug(case)}.png")
    return {"base_value": round(float(one.base_values), 4),
            "prediction": round(float(one.base_values + one.values.sum()), 4),
            "contributions": dict(sorted(((f, round(float(v), 4)) for f, v in zip(FEATURES, one.values)),
                                         key=lambda kv: -abs(kv[1])))}


# --- LIME --------------------------------------------------------------------------------------
def lime_explainer(train):
    # LIME's sampling statistics cannot handle NaN, so missing training values are median-filled
    # here (the same fill the pipeline's imputer uses).
    A = to_numeric(train)
    med = np.nanmedian(A, axis=0)
    A = np.where(np.isnan(A), med, A)
    return LimeTabularExplainer(
        A, mode="classification", feature_names=FEATURES, class_names=CLASS_NAMES,
        categorical_features=CAT_IDX, categorical_names={i: LEVELS[FEATURES[i]] for i in CAT_IDX},
        discretize_continuous=True, random_state=RANDOM_STATE), med


def local_lime(explainer, med, model, row: pd.DataFrame, cls: int, case: str):
    x = to_numeric(row)[0]
    missing = np.isnan(x)
    f = proba_fn(model)

    def predict(A):
        A = A.copy()
        A[:, missing] = np.nan   # the model still sees these as missing (imputed inside the pipeline)
        return f(A)

    exp = explainer.explain_instance(np.where(missing, med, x), predict, labels=(cls,),
                                     num_features=len(FEATURES), num_samples=5000)
    # as_map and as_list share one order; a held-missing feature gets an honest label, not its median bin
    rules = [f"{FEATURES[i]} = missing" if missing[i] else rule
             for (i, _), (rule, _) in zip(exp.as_map()[cls], exp.as_list(label=cls))]
    weights = [w for _, w in exp.as_list(label=cls)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(rules[::-1], weights[::-1], height=0.6,
            color=[CLASS_COLORS[cls] if w > 0 else "#a3a29c" for w in weights[::-1]])
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_xlabel(f"LIME weight (change in P({CLASS_NAMES[cls]}))")
    ax.set_title(f"LIME: local surrogate for P({CLASS_NAMES[cls]}), {case}  (local R² = {exp.score:.2f})",
                 fontsize=11)
    save(fig, f"lime_{slug(case)}.png")
    return {"intercept": round(float(exp.intercept[cls]), 4),
            "local_r2": round(float(exp.score), 4),
            "weights": {rule: round(float(w), 4) for rule, w in zip(rules, weights)},
            "missing_features_held_missing": [FEATURES[i] for i in np.where(missing)[0]]}


def slug(case: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in case.lower()).strip("_")


def pick_cases(test, model):
    """The inference-demo hold-out row + one tier-change month that persistence gets wrong."""
    full = pd.read_csv(PROCESSED_DIR / "model_dataset.csv")   # load_split() drops the hold-out row
    hold = full[full.is_holdout]
    cases = {f"{hold.plant_id.iloc[0]} {int(hold.year.iloc[0])}-{int(hold.month.iloc[0]):02d}": hold}

    persist = test.cf_lag1.apply(lambda v: cf_to_class(v) if pd.notna(v) else -1)
    pred = model.predict(test[FEATURES])
    hit = test[(persist != test[TARGET]) & (pred == test[TARGET]) & test.cf_lag1.notna()
               & (test.category == "Mini Hydro") & (test.month == 11)]
    if len(hit):
        r = hit.sample(1, random_state=RANDOM_STATE)
        cases[f"{r.plant_id.iloc[0]} {int(r.year.iloc[0])}-{int(r.month.iloc[0]):02d}"] = r
    return cases


def main():
    with open(MODELS_DIR / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    _, train, test = load_split()

    print("SHAP global (permutation explainer on the full pipeline) ...")
    sx = shap_explainer(model, train)
    out = {"model": "Soft Voting (RF+XGB), models/model.pkl",
           "shap_mean_abs_by_class": global_shap(sx, test), "cases": {}}

    lx, med = lime_explainer(train)
    for case, row in pick_cases(test, model).items():
        row = row[FEATURES].reset_index(drop=True)
        proba = model.predict_proba(row)[0]
        cls = int(proba.argmax())
        print(f"explaining {case}: predicted {CLASS_NAMES[cls]} ({proba[cls]:.3f}) ...")
        out["cases"][case] = {
            "inputs": dict(zip(FEATURES, display_values(row.iloc[0]))),
            "predicted_class": CLASS_NAMES[cls],
            "probabilities": dict(zip(CLASS_NAMES, np.round(proba, 4).tolist())),
            "shap": local_shap(sx, row, cls, case),
            "lime": local_lime(lx, med, model, row, cls, case),
        }

    (REPORTS_DIR / "explanations.json").write_text(json.dumps(out, indent=2))
    for case, c in out["cases"].items():
        top_shap = list(c["shap"]["contributions"].items())[:4]
        top_lime = list(c["lime"]["weights"].items())[:4]
        print(f"\n{case} -> {c['predicted_class']}\n  SHAP top: {top_shap}\n  LIME top: {top_lime}")
    print(f"\nfigures in reports/figures/, numbers in {REPORTS_DIR / 'explanations.json'}")


if __name__ == "__main__":
    main()
