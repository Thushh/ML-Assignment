"""FINAL ASSIGNMENT - interpreting the models (requirement 6).

Run:  python -m src.explain        (after src.final_train)

SHAP (SHapley Additive exPlanations) splits ONE prediction into a contribution per feature:
    model output for a class = base value + sum of the features' SHAP values
A positive SHAP value pushes the prediction TOWARDS that class, a negative one away from it.
CatBoost and LightGBM compute exact tree SHAP values natively (no extra package needed).

Outputs (reports/final/):
    explain.json
    figures/e1_shap_global.png        which features matter, per class (CatBoost) + LightGBM comparison
    figures/e2_shap_low_class.png     how feature VALUES push towards "Low" (under-delivery risk)
    figures/e3_holdout_explained.png  why the held-out plant-month got its prediction
    figures/e4_hybrid_weights.png     how the hybrid stack combines Markov / LightGBM / RF
"""
import json
import pickle

import numpy as np
import pandas as pd
from catboost import Pool

from .config import (CLASS_COLORS, CLASS_NAMES, COLORS, FINAL_FIG_DIR, FINAL_INPUT, FINAL_MODELS_DIR,
                     FINAL_REPORTS_DIR, PROCESSED_DIR, TARGET)
from .plotting import plt, save
from .train import load_split


def _save(fig, name):
    save(fig, name, folder=FINAL_FIG_DIR)


def _load(name):
    with open(FINAL_MODELS_DIR / name, "rb") as fh:
        return pickle.load(fh)


def catboost_shap(pipe, X):
    """Returns SHAP array (n_rows, 3 classes, n_features) and the transformed frame."""
    Z = pipe.named_steps["frame"].transform(X)
    cb = pipe.named_steps["model"]
    pool = Pool(Z, cat_features=list(cb.cat_features))
    sv = cb.model_.get_feature_importance(pool, type="ShapValues")   # (n, classes, features + 1 bias)
    return sv[:, :, :-1], sv[:, :, -1], Z


def lightgbm_shap(pipe, X):
    Z = pipe.named_steps["frame"].transform(X)
    contrib = pipe.named_steps["model"].booster_.predict(Z, pred_contrib=True)   # (n, 3 * (p + 1))
    p = Z.shape[1]
    sv = contrib.reshape(len(Z), 3, p + 1)
    return sv[:, :, :-1], Z


def fig_global(cb_sv, cb_cols, lgb_sv, lgb_cols, out):
    cb_imp = pd.DataFrame(np.abs(cb_sv).mean(axis=0).T, index=cb_cols, columns=CLASS_NAMES)
    order = cb_imp.sum(axis=1).sort_values().index
    lgb_imp = pd.Series(np.abs(lgb_sv).mean(axis=(0, 1)), index=lgb_cols)
    out["catboost_mean_abs_shap"] = cb_imp.loc[order[::-1]].round(4).to_dict("index")
    out["lightgbm_mean_abs_shap"] = lgb_imp.sort_values(ascending=False).round(4).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
    left = np.zeros(len(order))
    for k, c in enumerate(CLASS_NAMES):
        axes[0].barh(order, cb_imp.loc[order, c], left=left, color=CLASS_COLORS[k], label=c, height=0.65,
                     edgecolor=COLORS["surface"], linewidth=1.5)
        left += cb_imp.loc[order, c].values
    axes[0].set_title("CatBoost: mean |SHAP| per feature, split by class")
    axes[0].set_xlabel("Mean |SHAP| (log-odds), summed over classes")
    axes[0].legend(loc="lower right")
    l_order = lgb_imp.sort_values()
    axes[1].barh(l_order.index, l_order.values, color=COLORS["orange"], height=0.65)
    axes[1].set_title("LightGBM: mean |SHAP| (no plant_id)")
    axes[1].set_xlabel("Mean |SHAP| over the 3 classes")
    _save(fig, "e1_shap_global.png")
    return order[::-1]


def fig_low_class(cb_sv, Z, features):
    """Beeswarm-style plot for class 'Low': each dot = one test plant-month."""
    top = [f for f in features if f != "plant_id"][:8]
    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(0)
    cols = list(Z.columns)
    for i, f in enumerate(top[::-1]):
        sv = cb_sv[:, 0, cols.index(f)]
        val = Z[f]
        if val.dtype == object:
            color = COLORS["ink2"]
        else:
            v = val.astype(float).values
            lo, hi = np.nanpercentile(v, [5, 95])
            color = plt.cm.coolwarm(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))
        ax.scatter(sv, i + rng.uniform(-0.3, 0.3, len(sv)), s=6, c=color, alpha=0.6)
    ax.axvline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_yticks(range(len(top)), top[::-1])
    ax.set_xlabel("SHAP value for class 'Low' (right = pushes towards under-delivery)")
    ax.set_title("What drives a 'Low' prediction? (colour: blue = low feature value, red = high)")
    _save(fig, "e2_shap_low_class.png")


def fig_holdout(cb_pipe, out):
    hold = json.loads((PROCESSED_DIR / "holdout_sample.json").read_text())
    x = pd.DataFrame([{c: hold.get(c) if c != "plant_id" else hold["plant_id"] for c in FINAL_INPUT}])
    sv, base, Z = catboost_shap(cb_pipe, x)
    proba = cb_pipe.predict_proba(x)[0]
    k = int(proba.argmax())
    contrib = pd.Series(sv[0, k], index=Z.columns).sort_values(key=np.abs, ascending=False)
    top = contrib.head(9)[::-1]
    out["holdout"] = {"plant": hold["plant_id"], "month": f"{hold['year']}-{hold['month']:02d}",
                      "true_class": hold["true_class"], "predicted_class": CLASS_NAMES[k],
                      "probabilities": dict(zip(CLASS_NAMES, np.round(proba, 4).tolist())),
                      "top_contributions": contrib.head(9).round(4).to_dict()}

    def fmt(f):
        v = Z[f].iloc[0]
        return f"{f} = {v}" if isinstance(v, str) else f"{f} = {float(v):.3f}" if pd.notna(v) else f"{f} = missing"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh([fmt(f) for f in top.index], top.values,
            color=[COLORS["aqua"] if v > 0 else COLORS["orange"] for v in top.values], height=0.6)
    ax.axvline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_xlabel(f"SHAP contribution towards '{CLASS_NAMES[k]}' (log-odds)")
    ax.set_title(f"{hold['plant_id']}, {hold['year']}-{hold['month']:02d}: CatBoost predicts {CLASS_NAMES[k]} "
                 f"(P={proba[k]:.2f}); actual {hold['true_class']}")
    _save(fig, "e3_holdout_explained.png")


def fig_hybrid(stack, out):
    meta = stack.final_estimator_
    names = [f"{m} P({c})" for m in ("Markov", "LightGBM", "RF") for c in CLASS_NAMES]
    coef = pd.DataFrame(meta.coef_, index=CLASS_NAMES, columns=names)
    # influence of each base model = mean |coefficient| over its 3 inputs and the 3 classes
    influence = {m: float(np.abs(coef.iloc[:, i * 3:(i + 1) * 3]).values.mean())
                 for i, m in enumerate(["Markov chain", "LightGBM", "Random Forest"])}
    out["hybrid_meta_coefficients"] = coef.round(3).to_dict("index")
    out["hybrid_base_model_influence"] = {k: round(v, 3) for k, v in influence.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.4, 1]})
    vmax = np.abs(coef.values).max()
    axes[0].imshow(coef.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].grid(False)
    for i in range(3):
        for j in range(9):
            axes[0].text(j, i, f"{coef.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    axes[0].set_xticks(range(9), names, rotation=35, ha="right", fontsize=8)
    axes[0].set_yticks(range(3), [f"→ {c}" for c in CLASS_NAMES])
    axes[0].set_title("Hybrid stack: logistic-regression weights on each base model's probabilities")
    s = pd.Series(influence).sort_values()
    axes[1].barh(s.index, s.values, color=["#52514e", COLORS["orange"], COLORS["blue"]][:len(s)], height=0.55)
    axes[1].set_title("Average influence of each base model")
    _save(fig, "e4_hybrid_weights.png")


def main():
    _, _, test = load_split()
    X = test[FINAL_INPUT]
    out = {"note": "SHAP values of the deployment models (trained Feb-2023..Dec-2024) computed on Jul-Dec 2024 rows; they explain what the models learned"}
    cb, lgb, stack = _load("catboost.pkl"), _load("lightgbm.pkl"), _load("hybrid_stack.pkl")

    cb_sv, _, Zc = catboost_shap(cb, X)
    lgb_sv, Zl = lightgbm_shap(lgb, X)
    order = fig_global(cb_sv, list(Zc.columns), lgb_sv, list(Zl.columns), out)
    fig_low_class(cb_sv, Zc, list(order))
    fig_holdout(cb, out)
    fig_hybrid(stack, out)

    (FINAL_REPORTS_DIR / "explain.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("holdout", "hybrid_base_model_influence")}, indent=2, default=float))
    print("top CatBoost features:", list(out["catboost_mean_abs_shap"])[:6])


if __name__ == "__main__":
    main()
