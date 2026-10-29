"""Step 4 - explainability for the main model (NGBoost): its own feature importances, SHAP, LIME and PDP.

Run:  python -m src.explain      (after src.train)

NGBoost predicts two things for every plant-month: the EXPECTED capacity factor (mean) and how
UNCERTAIN it is (standard deviation). Both are explained.

Outputs (outputs/):
    ngboost_importance.png        which features drive the expected output vs the uncertainty (NGBoost's own)
    shap_global.png               mean |SHAP| per feature, for the mean and for the uncertainty (exact TreeSHAP)
    shap_beeswarm_mean.png        how each feature's value pushes the expected capacity factor up or down
    shap_local_holdout.png        exact breakdown of one prediction (the left-out plant-month)
    lime_holdout.png              LIME local surrogate for the same prediction
    pdp.png                       partial dependence of P(Low) and P(High)
    explanations.json             the numbers behind the figures
"""
import json
import pickle

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import partial_dependence

from .preprocess import (CATEGORICAL_FEATURES, CATEGORIES, CLASS_COLORS, CLASS_NAMES, COLORS, FEATURES, MODELS_DIR,
                         OUTPUTS_DIR, PROCESSED_DIR, RANDOM_STATE, SEASON_BY_MONTH, load_split, original_feature, plt,
                         save_fig)

SEASONS = list(dict.fromkeys(SEASON_BY_MONTH.values()))
LEVELS = {"category": CATEGORIES, "season": SEASONS}
CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]


def to_numeric(X):
    """LIME needs a numeric matrix: text categories -> integer codes."""
    X = X[FEATURES].copy()
    for c, levels in LEVELS.items():
        X[c] = X[c].map({v: i for i, v in enumerate(levels)})
    return X.to_numpy(dtype=float)


def to_frame(A):
    X = pd.DataFrame(np.atleast_2d(A), columns=FEATURES)
    for c, levels in LEVELS.items():
        X[c] = [levels[int(round(v))] for v in X[c]]
    return X


def holdout_row():
    hold = json.loads((PROCESSED_DIR / "holdout_sample.json").read_text())
    return hold, pd.DataFrame([{f: hold[f] for f in FEATURES}]).astype({"category": str, "season": str})


def per_feature(values, columns):
    """Sum encoded columns (one-hot levels, missing flags) back into the 13 original features.
    SHAP values are additive, so the sum is still an exact contribution."""
    groups = [original_feature(c) for c in columns]
    df = pd.DataFrame(np.atleast_2d(values), columns=groups)
    return df.T.groupby(level=0, sort=False).sum().T[FEATURES]


# ------------------------------------------------------------------------------ NGBoost's own importances
def ngboost_importance(model, out):
    cols = model.prep_.get_feature_names_out()
    imp = per_feature(model.ngb_.feature_importances_, cols)
    imp.index = ["expected output (mean)", "uncertainty (sd)"]
    out["ngboost_importance"] = imp.T.round(4).to_dict("index")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    order = imp.iloc[0].sort_values().index
    for ax, (label, row), c in zip(axes, imp.iterrows(), [COLORS["aqua"], COLORS["orange"]]):
        ax.barh(order, row[order], color=c, height=0.6)
        ax.set_title(f"NGBoost: what drives the {label}", fontsize=11)
        ax.set_xlabel("Share of split improvement")
    save_fig(fig, "ngboost_importance.png")


# ------------------------------------------------------------------------------ SHAP (exact TreeSHAP)
def shap_explain(model, test, hold, x, out):
    cols = model.prep_.get_feature_names_out()
    Xt = model.prep_.transform(test[FEATURES])
    mean_exp = shap.TreeExplainer(model.ngb_, model_output=0)     # explains the predicted mean CF
    sd_exp = shap.TreeExplainer(model.ngb_, model_output=1)       # explains log(sd), the uncertainty
    sv_mean, sv_sd = per_feature(mean_exp.shap_values(Xt), cols), per_feature(sd_exp.shap_values(Xt), cols)

    imp = pd.DataFrame({"expected output (mean)": sv_mean.abs().mean(), "uncertainty (log sd)": sv_sd.abs().mean()})
    imp = imp.sort_values("expected output (mean)")
    out["shap_mean_abs"] = imp.iloc[::-1].round(4).to_dict("index")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, col, c in zip(axes, imp.columns, [COLORS["aqua"], COLORS["orange"]]):
        ax.barh(imp.index, imp[col], color=c, height=0.6)
        ax.set_title(f"SHAP: influence on the {col}", fontsize=11)
        ax.set_xlabel("Mean |SHAP value|")
    save_fig(fig, "shap_global.png")

    data = test[FEATURES].copy()
    data[CATEGORICAL_FEATURES] = np.nan                            # category codes have no order -> drawn grey
    expl = shap.Explanation(sv_mean.values, base_values=np.full(len(sv_mean), mean_exp.expected_value),
                            data=data.astype(float).values, feature_names=FEATURES)
    plt.figure(figsize=(9, 5.5))
    shap.plots.beeswarm(expl, max_display=13, show=False, plot_size=None)
    plt.title("SHAP: how feature values move the expected capacity factor", loc="left")
    save_fig(plt.gcf(), "shap_beeswarm_mean.png")

    # one prediction, exactly decomposed: base value + contributions = predicted mean CF
    local = per_feature(mean_exp.shap_values(model.prep_.transform(x)), cols).iloc[0]
    mu, sd = model.predict_dist(x)
    proba = model.predict_proba(x)[0]
    k = int(proba.argmax())
    top = local.reindex(local.abs().sort_values(ascending=False).index)[:9][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(top.index, top.values * 100, color=[COLORS["aqua"] if v > 0 else COLORS["orange"] for v in top.values], height=0.6)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlabel("Contribution to the expected capacity factor (percentage points)")
    ax.set_title(f"{hold['plant_id']}, {hold['year']}-{hold['month']:02d}: expected CF {mu[0]:.0%} ± {sd[0]:.0%} "
                 f"-> {CLASS_NAMES[k]} (P={proba[k]:.2f}); actual CF {hold['cf']:.0%}", fontsize=10)
    save_fig(fig, "shap_local_holdout.png")
    out["holdout"] = {"plant": hold["plant_id"], "month": f"{hold['year']}-{hold['month']:02d}",
                      "actual_cf": round(hold["cf"], 3), "actual_tier": hold["true_class"],
                      "expected_cf": round(float(mu[0]), 3), "sd_cf": round(float(sd[0]), 3),
                      "predicted_tier": CLASS_NAMES[k], "probabilities": dict(zip(CLASS_NAMES, np.round(proba, 3).tolist())),
                      "shap_base_value": round(float(mean_exp.expected_value), 3),
                      "shap_contributions": local.reindex(local.abs().sort_values(ascending=False).index).round(4).to_dict()}
    return k


# ------------------------------------------------------------------------------ LIME
def lime_explain(model, train, x, k, out):
    A = to_numeric(train)
    med = np.nanmedian(A, axis=0)
    explainer = LimeTabularExplainer(np.where(np.isnan(A), med, A), mode="classification", feature_names=FEATURES,
                                     class_names=CLASS_NAMES, categorical_features=CAT_IDX,
                                     categorical_names={i: LEVELS[FEATURES[i]] for i in CAT_IDX},
                                     discretize_continuous=True, random_state=RANDOM_STATE)
    row = to_numeric(x)[0]
    missing = np.isnan(row)

    def predict(B):                          # keep originally-missing inputs missing, as the model saw them
        B = B.copy()
        B[:, missing] = np.nan
        return model.predict_proba(to_frame(B))

    exp = explainer.explain_instance(np.where(missing, med, row), predict, labels=(k,), num_features=len(FEATURES), num_samples=5000)
    rules = [f"{FEATURES[i]} = missing" if missing[i] else r for (i, _), (r, _) in zip(exp.as_map()[k], exp.as_list(label=k))]
    weights = [w for _, w in exp.as_list(label=k)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(rules[::-1], weights[::-1], color=[CLASS_COLORS[k] if w > 0 else COLORS["grey"] for w in weights[::-1]], height=0.6)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlabel(f"LIME weight (change in P({CLASS_NAMES[k]}))")
    ax.set_title(f"LIME for the hold-out prediction (local R² = {exp.score:.2f})")
    save_fig(fig, "lime_holdout.png")
    out["lime_holdout"] = {"local_r2": round(float(exp.score), 4), "weights": dict(zip(rules, np.round(weights, 4).tolist()))}


# ------------------------------------------------------------------------------ PDP
def pdp(model, test, out):
    features = ["cf_lag1", "cf_lag12", "plant_mean_cf", "cf_trend"]
    X = test[FEATURES].sample(600, random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.6), sharey=True)
    out["pdp"] = {}
    for ax, f in zip(axes, features):
        r = partial_dependence(model, X, [f], response_method="predict_proba", grid_resolution=25, kind="average")
        grid = r["grid_values"][0]
        for k in (0, 2):
            ax.plot(grid, r["average"][k], color=CLASS_COLORS[k], label=f"P({CLASS_NAMES[k]})", linewidth=2)
        ax.set_title(f, fontsize=10)
        out["pdp"][f] = {"grid": np.round(grid, 3).tolist(), "p_low": np.round(r["average"][0], 3).tolist(),
                         "p_high": np.round(r["average"][2], 3).tolist()}
    axes[0].set_ylabel("Average predicted probability")
    axes[0].legend(fontsize=8)
    fig.suptitle("Partial dependence: average effect of one feature, others held as observed", x=0.01, ha="left", fontweight="bold")
    save_fig(fig, "pdp.png")


def main():
    with open(MODELS_DIR / "ngboost.pkl", "rb") as fh:
        model = pickle.load(fh)
    train, val, test = load_split()
    hold, x = holdout_row()
    out = {"model": "NGBoost (models/ngboost.pkl), trained Feb-2023 .. Jun-2024"}

    print("NGBoost feature importances ...")
    ngboost_importance(model, out)
    print("SHAP (exact TreeSHAP) ...")
    k = shap_explain(model, test, hold, x, out)
    print("LIME ...")
    lime_explain(model, pd.concat([train, val]), x, k, out)
    print("partial dependence ...")
    pdp(model, test, out)

    (OUTPUTS_DIR / "explanations.json").write_text(json.dumps(out, indent=2, default=float))
    print("top SHAP features (mean):", list(out["shap_mean_abs"])[:6])
    print("hold-out:", json.dumps(out["holdout"], indent=2))


if __name__ == "__main__":
    main()
