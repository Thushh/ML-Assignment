"""Train, tune-check, evaluate and export RF, XGBoost, Soft Voting and Stacking.

Run:  python -m src.train
"""
import json
import pickle
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)

from .config import (CLASS_COLORS, CLASS_NAMES, COLORS, FEATURES, MODELS_DIR, PROCESSED_DIR,
                     REPORTS_DIR, TARGET, TRAIN_END_T)
from .features import cf_to_class
from .models import RF_PARAMS, XGB_PARAMS, build_pipelines, make_preprocessor, rf_model, xgb_model
from .plotting import plt, save

SLUGS = {"Random Forest": "random_forest", "XGBoost": "xgboost",
         "Soft Voting (RF+XGB)": "model", "Stacking (RF+XGB -> LogReg)": "stacking"}
MODEL_COLORS = {"Persistence baseline": "#a3a29c", "Random Forest": COLORS["blue"],
                "XGBoost": COLORS["orange"], "Soft Voting (RF+XGB)": COLORS["aqua"],
                "Stacking (RF+XGB -> LogReg)": "#4a3aa7"}


def load_split():
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv")
    df = df[~df.is_holdout].sort_values(["t", "plant_id"])
    train, test = df[df.t <= TRAIN_END_T], df[df.t > TRAIN_END_T]
    return df, train, test


def time_folds(train):
    """Expanding-window CV inside the training period: validate on 2-month blocks."""
    folds = []
    for start in (13, 15, 17):
        tr = np.where(train.t.values < start)[0]
        va = np.where((train.t.values >= start) & (train.t.values < start + 2))[0]
        folds.append((tr, va))
    return folds


def metrics(y, pred, proba):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "log_loss": log_loss(y, proba / proba.sum(1, keepdims=True), labels=[0, 1, 2])
        if proba is not None else np.nan,
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr") if proba is not None else np.nan,
    }


def persistence_baseline(X):
    """'Next month looks like last month' - the naive rule a planner would use."""
    return X.cf_lag1.fillna(X.cf_roll3).fillna(0.2).apply(cf_to_class).values


def validation_curves(train):
    """Sanity-check the hand-designed depth choices with time-ordered CV."""
    X, y = train[FEATURES], train[TARGET]
    folds = time_folds(train)
    from sklearn.pipeline import Pipeline

    def cv_f1(est):
        scores = []
        for tr, va in folds:
            p = Pipeline([("prep", make_preprocessor()), ("model", clone(est))])
            p.fit(X.iloc[tr], y.iloc[tr])
            scores.append(f1_score(y.iloc[va], p.predict(X.iloc[va]), average="macro"))
        return np.mean(scores)

    rf_depths = [4, 8, 14, 20, 30]
    xgb_depths = [2, 3, 5, 7, 9]
    rf_scores = [cv_f1(rf_model().set_params(max_depth=d, n_estimators=200)) for d in rf_depths]
    xgb_scores = [cv_f1(xgb_model().set_params(max_depth=d)) for d in xgb_depths]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, xs, ys, chosen, name, c in [
            (axes[0], rf_depths, rf_scores, RF_PARAMS["max_depth"], "Random Forest", COLORS["blue"]),
            (axes[1], xgb_depths, xgb_scores, XGB_PARAMS["max_depth"], "XGBoost", COLORS["orange"])]:
        ax.plot(xs, ys, color=c, marker="o", markersize=8)
        ax.axvline(chosen, color=COLORS["ink2"], linestyle="--", linewidth=1)
        ax.text(chosen, min(ys), f" chosen = {chosen}", color=COLORS["ink2"], fontsize=9)
        ax.set_title(f"{name}: max_depth vs time-CV macro-F1")
        ax.set_xlabel("max_depth")
    axes[0].set_ylabel("Mean macro-F1 (3 time folds)")
    save(fig, "validation_curves_max_depth.png")
    return {"rf_max_depth": dict(zip(map(str, rf_depths), rf_scores)),
            "xgb_max_depth": dict(zip(map(str, xgb_depths), xgb_scores))}


def plot_confusions(results, y_test):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8))
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, results[name]["pred"], labels=[0, 1, 2], normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.55 else COLORS["ink"])
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_title(name.replace(" (RF+XGB -> LogReg)", "").replace(" (RF+XGB)", ""), fontsize=11)
    axes[0].set_ylabel("Actual")
    save(fig, "confusion_matrices.png")


def plot_comparison(table):
    cols = ["accuracy", "macro_f1", "balanced_accuracy"]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.16
    for i, name in enumerate(table.index):
        xs = np.arange(len(cols)) + (i - (len(table) - 1) / 2) * width
        ax.bar(xs, table.loc[name, cols], width=width - 0.02, color=MODEL_COLORS[name], label=name)
    ax.set_xticks(range(len(cols)), ["Accuracy", "Macro-F1", "Balanced accuracy"])
    ax.set_ylim(0.4, 1)
    ax.set_title("Test performance (Jul-Dec 2024, unseen months)")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    save(fig, "model_comparison.png")


def regime_analysis(results, y_te, persist):
    """Split test rows into 'stable' (tier same as last month) and 'changed' months.

    Persistence is right on every stable row and wrong on every changed row by
    definition, so the changed rows show what the ML models add.
    """
    changed = persist != y_te.values
    out = {}
    for name, r in results.items():
        out[name] = {"stable_acc": accuracy_score(y_te[~changed], r["pred"][~changed]),
                     "changed_acc": accuracy_score(y_te[changed], r["pred"][changed])}
    out = pd.DataFrame(out).T
    out["n_stable"], out["n_changed"] = int((~changed).sum()), int(changed.sum())

    fig, ax = plt.subplots(figsize=(10, 3.8))
    names = list(out.index)
    xs = np.arange(2)
    w = 0.16
    for i, n in enumerate(names):
        ax.bar(xs + (i - (len(names) - 1) / 2) * w, out.loc[n, ["stable_acc", "changed_acc"]],
               width=w - 0.02, color=MODEL_COLORS[n], label=n)
    ax.set_xticks(xs, [f"Stable months (n={out.n_stable.iloc[0]})",
                       f"Tier-change months (n={out.n_changed.iloc[0]})"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where the ensembles add value: months when a plant changes tier")
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    save(fig, "regime_change_accuracy.png")
    return out


def per_month(results, test):
    """Macro-F1 per test month: persistence vs the soft-voting ensemble."""
    rows = {}
    for t, g in test.groupby("t"):
        idx = (test.t == t).values
        rows[f"{2023 + (t - 1) // 12}-{(t - 1) % 12 + 1:02d}"] = {
            n: f1_score(test[TARGET][idx], results[n]["pred"][idx], average="macro")
            for n in ("Persistence baseline", "Soft Voting (RF+XGB)")}
    out = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for n in out.columns:
        ax.plot(out.index, out[n], marker="o", markersize=8, color=MODEL_COLORS[n], label=n)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Monthly test macro-F1: persistence vs soft-voting ensemble")
    ax.legend(loc="lower left")
    save(fig, "per_month_macro_f1.png")
    return out


def plot_importances(pipes, X_test, y_test):
    prep = pipes["Random Forest"].named_steps["prep"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    rf_imp = pd.Series(pipes["Random Forest"].named_steps["model"].feature_importances_, index=names)
    xgb_imp = pd.Series(pipes["XGBoost"].named_steps["model"].feature_importances_, index=names)
    perm = permutation_importance(pipes["Soft Voting (RF+XGB)"], X_test, y_test, n_repeats=8,
                                  scoring="f1_macro", random_state=0, n_jobs=1)
    perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, s, title, c in [(axes[0], rf_imp.nlargest(12).sort_values(), "Random Forest (impurity)", COLORS["blue"]),
                            (axes[1], xgb_imp.nlargest(12).sort_values(), "XGBoost (gain share)", COLORS["orange"]),
                            (axes[2], perm_imp, "Soft Voting (permutation, F1 drop)", COLORS["aqua"])]:
        ax.barh(s.index, s.values, color=c, height=0.6)
        ax.set_title(title, fontsize=11)
    save(fig, "feature_importance.png")
    return {"rf_impurity": rf_imp.sort_values(ascending=False).round(4).to_dict(),
            "xgb_gain": xgb_imp.sort_values(ascending=False).round(4).to_dict(),
            "voting_permutation": perm_imp.sort_values(ascending=False).round(4).to_dict()}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df, train, test = load_split()
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    print(f"train rows={len(train)}  test rows={len(test)}")

    print("validation curves ...")
    vc = validation_curves(train)

    pipes = build_pipelines()
    results = {"Persistence baseline": {"pred": persistence_baseline(X_te), "proba": None, "fit_s": 0.0}}
    cv_scores = {}
    folds = time_folds(train)
    for name, pipe in pipes.items():
        print(f"fitting {name} ...")
        fold_f1 = []
        for tr, va in folds:
            p = clone(pipe).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            fold_f1.append(f1_score(y_tr.iloc[va], p.predict(X_tr.iloc[va]), average="macro"))
        cv_scores[name] = {"mean": float(np.mean(fold_f1)), "std": float(np.std(fold_f1))}
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        results[name] = {"pred": pipe.predict(X_te), "proba": pipe.predict_proba(X_te),
                         "fit_s": time.time() - t0}

    table = pd.DataFrame({n: {**metrics(y_te, r["pred"], r["proba"]), "fit_seconds": r["fit_s"],
                              "time_cv_macro_f1": cv_scores.get(n, {}).get("mean", np.nan)}
                          for n, r in results.items()}).T
    table.to_csv(REPORTS_DIR / "metrics.csv", float_format="%.4f")
    print(table.round(4).to_string())

    # per-technology macro-F1 on the test months
    per_cat = {}
    for name in pipes:
        per_cat[name] = {c: f1_score(y_te[test.category == c], results[name]["pred"][(test.category == c).values],
                                     average="macro") for c in test.category.unique()}
    per_cat_df = pd.DataFrame(per_cat).round(3)
    per_cat_df["test_rows"] = test.category.value_counts()
    per_cat_df.to_csv(REPORTS_DIR / "per_category_macro_f1.csv")

    regime = regime_analysis(results, y_te, results["Persistence baseline"]["pred"])
    regime.to_csv(REPORTS_DIR / "regime_change_accuracy.csv", float_format="%.4f")
    print(regime.round(3).to_string())

    monthly = per_month(results, test)
    monthly.to_csv(REPORTS_DIR / "per_month_macro_f1.csv", float_format="%.4f")
    print(monthly.round(3).to_string())

    reports = {n: classification_report(y_te, r["pred"], target_names=CLASS_NAMES, output_dict=True)
               for n, r in results.items()}

    plot_confusions({n: r for n, r in results.items()}, y_te)
    plot_comparison(table)
    importances = plot_importances(pipes, X_te, y_te)

    best = table.drop(index="Persistence baseline").macro_f1.idxmax()
    for name, pipe in pipes.items():
        with open(MODELS_DIR / f"{SLUGS[name]}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)
    # the lecturer's soft-voting pipeline is the default model -> models/model.pkl

    meta = {
        "task": "3-class classification of next-month capacity-factor tier for Sri Lankan NCRE plants",
        "classes": CLASS_NAMES, "features": FEATURES,
        "train_period": "2023-02 .. 2024-06", "test_period": "2024-07 .. 2024-12",
        "train_rows": len(train), "test_rows": len(test),
        "default_model": "Soft Voting (RF+XGB)", "best_test_macro_f1_model": best,
        "model_files": {n: f"{s}.pkl" for n, s in SLUGS.items()},
        "test_metrics": table.round(4).to_dict("index"),
        "time_cv_macro_f1": cv_scores, "validation_curves": vc,
        "per_category_macro_f1": per_cat_df.drop(columns="test_rows").to_dict(),
        "classification_reports": reports,
        "per_month_macro_f1": monthly.round(4).to_dict("index"),
        "regime_change_accuracy": regime.round(4).to_dict("index"), "feature_importance": importances,
        "rf_oob_score": float(pipes["Random Forest"].named_steps["model"].oob_score_),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nbest by test macro-F1: {best}\nmodels saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
