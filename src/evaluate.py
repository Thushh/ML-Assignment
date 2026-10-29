"""Step 3 - evaluation on the untouched test months (Jul-Dec 2024); plots saved to outputs/.

Run:  python -m src.evaluate     (after src.train)

Outputs:
    outputs/metrics.csv                   all metrics, all models
    outputs/significance.csv              NGBoost vs each model: McNemar test + bootstrap CI
    outputs/per_technology_macro_f1.csv   macro-F1 by technology
    outputs/classification_report_ngboost.txt precision / recall / F1 per class
    outputs/ngboost_uncertainty.csv       do NGBoost's prediction intervals contain the actual CF as often as claimed?
    outputs/confusion_matrices.png, model_comparison.png, monthly_macro_f1.png
    outputs/ngboost_interval_calibration.png, ngboost_intervals_example.png
    outputs/holdout_prediction.json       single-sample inference on the left-out row
"""
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, roc_auc_score)

from .preprocess import (CLASS_NAMES, COLORS, FEATURES, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR,
                         RANDOM_STATE, TARGET, cf_to_class, class_gwh_band, load_split, plt, save_fig)

MAIN = "NGBoost"
MODEL_COLORS = {"Same as last month": COLORS["grey"], "Naive Bayes": "#c9a227", "Gaussian Process": "#7b5ea7",
                "EBM": "#52514e", "Random Forest": COLORS["blue"], "XGBoost": COLORS["orange"], "NGBoost": COLORS["aqua"]}
COMPARED = ["Naive Bayes", "Gaussian Process", "EBM", "Random Forest", "XGBoost", "NGBoost"]


def persistence(X):
    """Baseline rule: next month's tier = last month's tier."""
    return X.cf_lag1.fillna(X.cf_roll3).fillna(0.2).apply(cf_to_class).values


def load_models():
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    models = {}
    for name, f in meta["model_files"].items():
        with open(MODELS_DIR / f, "rb") as fh:
            models[name] = pickle.load(fh)
    return models


def metrics(y, pred, proba, changed):
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "low_f1": f1_score(y, pred, labels=[0], average="macro"),
            "tier_change_accuracy": accuracy_score(y[changed], pred[changed]),
            "log_loss": log_loss(y, proba, labels=[0, 1, 2]) if proba is not None else np.nan,
            "roc_auc": roc_auc_score(y, proba, multi_class="ovr") if proba is not None else np.nan}


def significance(y, preds, ref=MAIN, n_boot=1000):
    """NGBoost vs each model: exact McNemar test on disagreements + bootstrap 95 % CI of the macro-F1 difference."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for name, p in preds.items():
        if name == ref:
            continue
        a, b = preds[ref] == y, p == y
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        diffs = [f1_score(y[i], preds[ref][i], average="macro") - f1_score(y[i], p[i], average="macro") for i in idx]
        rows.append({"vs": name, "macro_f1_difference": f1_score(y, preds[ref], average="macro") - f1_score(y, p, average="macro"),
                     "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
                     "ngboost_right_other_wrong": n01, "ngboost_wrong_other_right": n10,
                     "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0})
    return pd.DataFrame(rows).round(4)


def plot_confusions(y, preds):
    preds = {n: preds[n] for n in ["Same as last month", "Gaussian Process", "XGBoost", MAIN]}
    fig, axes = plt.subplots(1, len(preds), figsize=(3.6 * len(preds), 3.6))
    for ax, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y, p, labels=[0, 1, 2], normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.55 else COLORS["ink"])
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("Actual")
    save_fig(fig, "confusion_matrices.png")


def plot_comparison(table):
    cols = ["accuracy", "macro_f1", "low_f1", "tier_change_accuracy"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.13
    for i, name in enumerate(table.index):
        xs = np.arange(len(cols)) + (i - (len(table) - 1) / 2) * w
        ax.bar(xs, table.loc[name, cols], width=w - 0.02, color=MODEL_COLORS[name], label=name)
    ax.set_xticks(range(len(cols)), ["Accuracy", "Macro-F1", "Low-tier F1", "Tier-change accuracy"])
    ax.set_ylim(0, 1)
    ax.set_title("Test performance, Jul-Dec 2024 (unseen months)", pad=48)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save_fig(fig, "model_comparison.png")


def plot_monthly(test, preds):
    months = sorted(test.t.unique())
    labels = [f"{2023 + (t - 1) // 12}-{(t - 1) % 12 + 1:02d}" for t in months]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name in ["Same as last month", "Gaussian Process", "XGBoost", MAIN]:
        scores = [f1_score(test[TARGET][test.t == t], preds[name][(test.t == t).values], average="macro") for t in months]
        ax.plot(labels, scores, marker="o", color=MODEL_COLORS[name], label=name, linewidth=2)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 per test month")
    ax.legend()
    save_fig(fig, "monthly_macro_f1.png")


def uncertainty(model, test):
    """NGBoost predicts a distribution, so check its intervals: a 'P %' interval should contain the
    actual capacity factor P % of the time (coverage)."""
    X, cf = test[FEATURES], test["cf"].values
    rows = []
    for level in (0.5, 0.8, 0.9, 0.95):
        lo, hi = model.predict_interval(X, level)
        rows.append({"nominal": level, "actual_coverage": round(float(np.mean((cf >= lo) & (cf <= hi))), 3),
                     "mean_width_cf": round(float(np.mean(hi - lo)), 3)})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "ngboost_uncertainty.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0.4, 1], [0.4, 1], color=COLORS["grey"], linestyle="--", label="ideal")
    ax.plot(table.nominal, table.actual_coverage, marker="o", color=COLORS["aqua"], linewidth=2, label="NGBoost")
    ax.set_xlabel("Interval level the model claims")
    ax.set_ylabel("Share of test months inside it")
    ax.set_title("Are NGBoost's intervals honest?")
    ax.legend()
    save_fig(fig, "ngboost_interval_calibration.png")

    plant = test[test.Plant_Name == "Kirkoswald"].sort_values("t")
    mu, _ = model.predict_dist(plant[FEATURES])
    lo, hi = model.predict_interval(plant[FEATURES], 0.8)
    labels = [f"{2023 + (t - 1) // 12}-{(t - 1) % 12 + 1:02d}" for t in plant.t]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(labels, lo, hi, color=COLORS["aqua"], alpha=0.25, label="80 % interval")
    ax.plot(labels, mu, color=COLORS["aqua"], marker="o", linewidth=2, label="predicted mean")
    ax.plot(labels, plant.cf, color=COLORS["ink"], marker="s", linewidth=2, label="actual")
    ax.axhline(0.15, color=COLORS["orange"], linestyle=":", linewidth=1)
    ax.axhline(0.40, color=COLORS["blue"], linestyle=":", linewidth=1)
    ax.set_ylabel("Capacity factor")
    ax.set_title("Kirkoswald (mini-hydro), test months: predicted distribution vs actual")
    ax.legend(fontsize=8)
    save_fig(fig, "ngboost_intervals_example.png")
    return table


def holdout_check(models):
    """Single-sample inference on the plant-month kept out of every split."""
    hold = json.loads((PROCESSED_DIR / "holdout_sample.json").read_text())
    x = pd.DataFrame([{f: hold[f] for f in FEATURES}])
    out = {"plant": hold["plant_id"], "month": f"{hold['year']}-{hold['month']:02d}",
           "actual": f"{hold['generation_gwh']} GWh, CF {hold['cf']:.1%} -> {hold['true_class']}", "predictions": {}}
    for name, m in models.items():
        p = m.predict_proba(x)[0]
        k = int(p.argmax())
        out["predictions"][name] = {"class": CLASS_NAMES[k], "probabilities": dict(zip(CLASS_NAMES, np.round(p, 3).tolist())),
                                    "expected_generation": class_gwh_band(k, hold["installed_capacity_mw"], hold["year"], hold["month"])}
    (OUTPUTS_DIR / "holdout_prediction.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    _, _, test = load_split()
    X, y = test[FEATURES], test[TARGET].values
    models = load_models()
    rule = persistence(X)
    changed = rule != y                       # months where the tier changed from last month

    preds = {"Same as last month": rule}
    table = {"Same as last month": metrics(y, rule, None, changed)}
    for name in COMPARED:
        proba = models[name].predict_proba(X)
        preds[name] = proba.argmax(1)
        table[name] = metrics(y, preds[name], proba, changed)
    table = pd.DataFrame(table).T.round(4)
    table.to_csv(OUTPUTS_DIR / "metrics.csv")
    print(table.to_string())

    sig = significance(y, preds)
    sig.to_csv(OUTPUTS_DIR / "significance.csv", index=False)
    print("\n" + sig.to_string(index=False))

    per_tech = pd.DataFrame({name: {c: f1_score(y[(test.category == c).values], p[(test.category == c).values], average="macro")
                                    for c in test.category.unique()} for name, p in preds.items()}).round(3)
    per_tech["test_rows"] = test.category.value_counts()
    per_tech.to_csv(OUTPUTS_DIR / "per_technology_macro_f1.csv")
    (OUTPUTS_DIR / "classification_report_ngboost.txt").write_text(classification_report(y, preds[MAIN], target_names=CLASS_NAMES, digits=3))
    print("\nNGBoost interval coverage:\n" + uncertainty(models[MAIN], test).to_string(index=False))

    plot_confusions(y, preds)
    plot_comparison(table)
    plot_monthly(test, preds)
    print("\nhold-out:", json.dumps(holdout_check(models), indent=2))


if __name__ == "__main__":
    main()
