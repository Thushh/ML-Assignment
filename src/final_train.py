"""FINAL ASSIGNMENT - compare all models and find the best one for the use case.

Run:  python -m src.final_train      (about 12 minutes)

USE CASE
--------
CEB planners want to know, before a month starts, which renewable plants will deliver Low / Moderate /
High output. A useful model must therefore
  1. classify all three tiers well, including the minority "Low" (under-delivery) tier  -> macro-F1
  2. give trustworthy probabilities, so plants can be ranked by risk                     -> log-loss, ROC-AUC
  3. add value over "same as last month", i.e. catch tier changes                         -> tier-change accuracy

EVALUATION DESIGN - rolling-origin (protocol v3)
------------------------------------------------
Every month of 2024 (12 months, a full monsoon cycle) is predicted by models trained on ALL earlier months:
    Jan-24 <- trained on Feb-23..Dec-23,  Feb-24 <- Feb-23..Jan-24,  ...,  Dec-24 <- Feb-23..Nov-24
CatBoost/LightGBM were tuned on Jul-Dec 2023 only (src/tune.py), so no evaluation month was used for tuning.
The single hold-out row (Kirkoswald, Dec-2024) is excluded everywhere and kept for the inference demo.

SELECTION RULE (fixed before this run)
--------------------------------------
0. candidates = models that output probabilities (use-case need 2); rules are benchmarks
1. best = highest mean monthly macro-F1 over the 12 months
2. top group = candidates NOT significantly worse than the best
   (one-sided Wilcoxon signed-rank test on the 12 paired monthly macro-F1 values, p >= 0.05)
3. within the top group: lowest log-loss (best probabilities), then fastest to train
The selected model (and every other model) is then refitted on all data up to Dec-2024 for deployment.

Outputs
-------
models/final/*.pkl, models/final/final_metadata.json
reports/final/final_metrics.csv, monthly_macro_f1.csv, significance.csv,
reports/final/per_technology_macro_f1.csv, model_selection_report.md, figures/f1..f4
"""
import json
import pickle
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest, wilcoxon
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             recall_score, roc_auc_score)

from .config import (CLASS_NAMES, COLORS, EVAL_MONTHS, FINAL_FIG_DIR, FINAL_INPUT, FINAL_MODELS_DIR,
                     FINAL_REPORTS_DIR, N_BOOTSTRAP, RANDOM_STATE, SIGNIFICANCE_LEVEL, TARGET)
from .final_models import GROUPS, build_final_models, is_deterministic, previous_class
from .plotting import plt, save
from .train import load_split

GROUP_COLORS = {"Existing approach": "#a3a29c", "Activity model": COLORS["blue"], "New model": COLORS["aqua"]}
SLUGS = {
    "Persistence (same as last month)": "persistence",
    "Seasonal naive (same month last year)": "seasonal_naive",
    "Markov chain": "markov_chain",
    "Logistic regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Soft Voting (RF+XGB)": "soft_voting",
    "CatBoost": "catboost",
    "LightGBM": "lightgbm",
    "Hybrid stack (Markov+LightGBM+RF)": "hybrid_stack",
}
MONTH_LABEL = {t: f"{2023 + (t - 1) // 12}-{(t - 1) % 12 + 1:02d}" for t in range(1, 25)}


def _save(fig, name):
    save(fig, name, folder=FINAL_FIG_DIR)


def short(n):
    return (n.replace(" (same as last month)", "").replace(" (same month last year)", "")
             .replace(" (Markov+LightGBM+RF)", "").replace(" (RF+XGB)", ""))


def fast_macro_f1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec, rec = tp / cm.sum(0), tp / cm.sum(1)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(np.nanmean(f1))


# --------------------------------------------------------------------------- 1. rolling-origin evaluation
def rolling_evaluation(models, df):
    X, y, t = df[FINAL_INPUT], df[TARGET].values, df.t.values
    ev = np.isin(t, EVAL_MONTHS)
    prev = previous_class(df)
    results = {}
    for name, model in models.items():
        P = np.zeros((len(df), 3))
        monthly, fit_times = {}, []
        for m in EVAL_MONTHS:
            tr, te = np.where(t < m)[0], np.where(t == m)[0]
            t0 = time.time()
            fitted = clone(model).fit(X.iloc[tr], y[tr])
            fit_times.append(time.time() - t0)
            P[te] = fitted.predict_proba(X.iloc[te])
            monthly[MONTH_LABEL[m]] = f1_score(y[te], P[te].argmax(1), average="macro")
        yE, PE = y[ev], P[ev]
        pred = PE.argmax(1)
        det = is_deterministic(model)
        changed = prev[ev] != yE
        results[name] = {
            "pred": pred, "proba": PE, "monthly": monthly, "deterministic": det,
            "mean_monthly_macro_f1": float(np.mean(list(monthly.values()))),
            "std_monthly_macro_f1": float(np.std(list(monthly.values()))),
            "pooled_macro_f1": f1_score(yE, pred, average="macro"),
            "accuracy": accuracy_score(yE, pred),
            "balanced_accuracy": balanced_accuracy_score(yE, pred),
            "low_f1": f1_score(yE, pred, labels=[0], average="macro"),
            "low_recall": recall_score(yE, pred, labels=[0], average="macro"),
            "tier_change_acc": accuracy_score(yE[changed], pred[changed]),
            "stable_acc": accuracy_score(yE[~changed], pred[~changed]),
            "log_loss": np.nan if det else log_loss(yE, PE / PE.sum(1, keepdims=True), labels=[0, 1, 2]),
            "roc_auc": np.nan if det else roc_auc_score(yE, PE, multi_class="ovr"),
            "mean_fit_seconds": float(np.mean(fit_times)),
        }
        r = results[name]
        print(f"{name:40s} monthly macro-F1 {r['mean_monthly_macro_f1']:.4f} ± {r['std_monthly_macro_f1']:.4f} | "
              f"pooled {r['pooled_macro_f1']:.4f} | log-loss {r['log_loss']:.4f} | change-acc {r['tier_change_acc']:.3f}")
    return results, ev


# --------------------------------------------------------------------------- 2. selection rule
def select(results):
    cands = [n for n, r in results.items() if not r["deterministic"]]
    bench = [n for n in results if n not in cands]
    best = max(cands, key=lambda n: results[n]["mean_monthly_macro_f1"])
    b = np.array(list(results[best]["monthly"].values()))
    top, tests = [best], {}
    for n in cands:
        if n == best:
            continue
        o = np.array(list(results[n]["monthly"].values()))
        p = wilcoxon(b, o, alternative="greater").pvalue if np.any(b != o) else 1.0
        tests[n] = float(p)
        if p >= SIGNIFICANCE_LEVEL:
            top.append(n)
    selected = min(top, key=lambda n: (results[n]["log_loss"], results[n]["mean_fit_seconds"]))
    steps = [
        f"0. Candidates = {len(cands)} models with probability outputs; benchmarks only: {', '.join(bench)}.",
        f"1. Highest mean monthly macro-F1 (12 rolling months of 2024): {best} "
        f"({results[best]['mean_monthly_macro_f1']:.4f}).",
        f"2. Top group (not significantly worse than {best}, one-sided Wilcoxon p >= {SIGNIFICANCE_LEVEL}): "
        + ", ".join(f"{n}" + (f" (p={tests[n]:.3f})" if n in tests else "") for n in top) + ".",
        "   Excluded as significantly worse: "
        + (", ".join(f"{n} (p={p:.3f})" for n, p in tests.items() if p < SIGNIFICANCE_LEVEL) or "none") + ".",
        f"3. Lowest log-loss within the top group: SELECTED = {selected} "
        f"(log-loss {results[selected]['log_loss']:.4f}, macro-F1 {results[selected]['mean_monthly_macro_f1']:.4f}).",
    ]
    return selected, steps, top, tests


# --------------------------------------------------------------------------- 3. significance vs the selected model
def significance(results, selected, y):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(y), size=(N_BOOTSTRAP, len(y)))
    sp = results[selected]["pred"]
    s_ok = sp == y
    boot_s = np.array([fast_macro_f1(y[i], sp[i]) for i in idx])
    sm = np.array(list(results[selected]["monthly"].values()))
    rows = []
    for n, r in results.items():
        if n == selected:
            continue
        o_ok = r["pred"] == y
        b, c = int((s_ok & ~o_ok).sum()), int((~s_ok & o_ok).sum())
        mc = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        om = np.array(list(r["monthly"].values()))
        wx = wilcoxon(sm, om).pvalue if np.any(sm != om) else 1.0
        diff = boot_s - np.array([fast_macro_f1(y[i], r["pred"][i]) for i in idx])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"vs_model": n, "group": GROUPS[n],
                     "monthly_macro_f1_diff": float(sm.mean() - om.mean()), "wilcoxon_p": float(wx),
                     "pooled_macro_f1_diff": fast_macro_f1(y, sp) - fast_macro_f1(y, r["pred"]),
                     "ci95_low": lo, "ci95_high": hi, "mcnemar_p": float(mc),
                     "months_selected_better": int((sm > om).sum())})
    return pd.DataFrame(rows).sort_values("monthly_macro_f1_diff")


def per_technology(results, df_eval):
    y = df_eval[TARGET].values
    cats = df_eval.category.value_counts().index
    out = pd.DataFrame({n: {c: f1_score(y[df_eval.category.values == c], r["pred"][df_eval.category.values == c],
                                        average="macro") for c in cats} for n, r in results.items()})
    out["rows"] = df_eval.category.value_counts()
    return out


# --------------------------------------------------------------------------- figures
def fig_scorecard(table, selected, top):
    t = table.sort_values("mean_monthly_macro_f1")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(t))
    axes[0].barh(y, t.mean_monthly_macro_f1, xerr=t.std_monthly_macro_f1 / np.sqrt(len(EVAL_MONTHS)),
                 color=[GROUP_COLORS[g] for g in t.group], height=0.6,
                 error_kw={"ecolor": COLORS["ink2"], "elinewidth": 1, "capsize": 3})
    axes[0].set_yticks(y, [("★ " if n == selected else ("• " if n in top else "")) + short(n) for n in t.index])
    axes[0].set_xlim(0.45, 0.72)
    axes[0].set_xlabel("Mean monthly macro-F1, 12 rolling months of 2024 (± standard error)")
    axes[0].set_title("Classification quality  (★ selected, • top group)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[0].legend(handles, list(GROUP_COLORS), loc="lower right", fontsize=9)
    p = t[t.log_loss.notna()].sort_values("log_loss", ascending=False)
    axes[1].barh(np.arange(len(p)), p.log_loss, color=[GROUP_COLORS[g] for g in p.group], height=0.6)
    axes[1].set_yticks(np.arange(len(p)), [("★ " if n == selected else "") + short(n) for n in p.index])
    axes[1].set_xlim(p.log_loss.min() - 0.05, p.log_loss.max() + 0.02)
    axes[1].set_xlabel("Log-loss (lower = better probabilities)")
    axes[1].set_title("Probability quality")
    _save(fig, "f1_scorecard.png")


def fig_significance(sig, selected):
    s = sig.sort_values("pooled_macro_f1_diff")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(s))
    ax.errorbar(s.pooled_macro_f1_diff, y, xerr=[s.pooled_macro_f1_diff - s.ci95_low, s.ci95_high - s.pooled_macro_f1_diff],
                fmt="o", color=COLORS["blue"], ecolor=COLORS["blue"], elinewidth=2, capsize=4, markersize=7)
    ax.axvline(0, color=COLORS["ink2"], linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{short(n)}  (Wilcoxon p={w:.3f}, better in {k}/12 months)"
                      for n, w, k in zip(s.vs_model, s.wilcoxon_p, s.months_selected_better)])
    ax.set_xlabel(f"Macro-F1 of {short(selected)} minus the other model (pooled 2024, 95 % bootstrap CI)")
    ax.set_title(f"{short(selected)} vs every other model: right of 0 = {short(selected)} better")
    _save(fig, "f2_significance.png")


def fig_regime_monthly(table, monthly, selected):
    show = list(dict.fromkeys(["Persistence (same as last month)", "Markov chain", "Soft Voting (RF+XGB)",
                               "CatBoost", "LightGBM", selected]))
    colors = ["#a3a29c", "#52514e", COLORS["blue"], COLORS["aqua"], COLORS["orange"], "#4a3aa7"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    w = 0.8 / len(show)
    for i, n in enumerate(show):
        axes[0].bar(np.arange(2) + (i - (len(show) - 1) / 2) * w, table.loc[n, ["stable_acc", "tier_change_acc"]],
                    width=w - 0.02, color=colors[i % len(colors)], label=short(n))
        axes[1].plot(monthly.index, monthly[n], marker="o", markersize=6, color=colors[i % len(colors)],
                     label=short(n), linewidth=3 if n == selected else 1.4)
    axes[0].set_xticks(range(2), ["Stable months", "Tier-change months"])
    axes[0].set_ylabel("Accuracy (2024)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy when the tier stays vs changes")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Macro-F1")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Macro-F1 per month (each month predicted from all earlier months)")
    _save(fig, "f3_regime_and_monthly.png")


def fig_confusion_and_tech(results, per_cat, y, selected):
    cm = confusion_matrix(y, results[selected]["pred"], labels=[0, 1, 2], normalize="true")
    order = [n for n in results if n != "Seasonal naive (same month last year)"]
    pc = per_cat[order].T
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 2.2]})
    axes[0].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    axes[0].grid(False)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.55 else COLORS["ink"])
    axes[0].set_xticks(range(3), CLASS_NAMES)
    axes[0].set_yticks(range(3), CLASS_NAMES)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{short(selected)}: confusion, 2024 (row-normalised)")
    im = axes[1].imshow(pc.values, cmap="Blues", vmin=0.2, vmax=0.9, aspect="auto")
    axes[1].grid(False)
    for i in range(pc.shape[0]):
        for j in range(pc.shape[1]):
            v = pc.values[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.65 else COLORS["ink"])
    axes[1].set_xticks(range(pc.shape[1]), [f"{c}\n(n={per_cat.loc[c, 'rows']})" for c in pc.columns], fontsize=8)
    axes[1].set_yticks(range(pc.shape[0]), [short(n) for n in pc.index])
    axes[1].set_title("Macro-F1 by technology, 2024 (small n = noisy)")
    fig.colorbar(im, ax=axes[1], fraction=0.025)
    _save(fig, "f4_confusion_and_technology.png")


# --------------------------------------------------------------------------- report
def write_report(table, steps, selected, top, sig, monthly, best_params, n_eval):
    fmt = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"
    L = ["# Model selection report (final assignment)", "",
         "Generated by `src/final_train.py` (protocol v3).", "",
         "## 1. Use case and what 'best' means", "",
         "CEB planners need, before each month, the expected output tier (Low / Moderate / High) of every renewable plant.",
         "A good model must (1) classify all tiers, including the minority **Low** tier (under-delivery) -> **macro-F1**; "
         "(2) give trustworthy **probabilities** to rank plants by risk -> **log-loss**, ROC-AUC; "
         "(3) beat 'same as last month' where it matters -> **tier-change accuracy**.", "",
         "## 2. Evaluation design: rolling origin over 2024", "",
         f"Each month of 2024 is predicted by models trained on **all earlier months** ({n_eval:,} predictions, 12 months, "
         "a full monsoon cycle). CatBoost/LightGBM were tuned on Jul-Dec 2023 only. The hold-out row (Kirkoswald, Dec-2024) "
         "is excluded everywhere.", "",
         "## 3. Results (2024, rolling origin)", "",
         "| Group | Model | Monthly macro-F1 (mean ± std) | Pooled macro-F1 | Accuracy | Low F1 | Tier-change acc | ROC-AUC | Log-loss | Fit (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in table.sort_values("mean_monthly_macro_f1", ascending=False).iterrows():
        mark = "**" if n == selected else ""
        L.append(f"| {r.group} | {mark}{n}{mark}{' •' if n in top and n != selected else ''} | "
                 f"{r.mean_monthly_macro_f1:.3f} ± {r.std_monthly_macro_f1:.3f} | {r.pooled_macro_f1:.3f} | {r.accuracy:.3f} | "
                 f"{r.low_f1:.3f} | {r.tier_change_acc:.3f} | {fmt(r.roc_auc)} | {fmt(r.log_loss)} | {r.mean_fit_seconds:.1f} |")
    L += ["", "**Bold** = selected, • = in the top group.", "",
          "## 4. Selection (rule fixed before this run)", ""] + [f"* {s}" for s in steps]
    L += ["", f"## 5. Is {selected} really better? Paired tests on 2024", "",
          "| vs model | monthly macro-F1 diff | Wilcoxon p (12 months) | months better | pooled diff | 95 % bootstrap CI | McNemar p |",
          "|---|---|---|---|---|---|---|"]
    for _, r in sig.sort_values("monthly_macro_f1_diff").iterrows():
        L.append(f"| {r.vs_model} | {r.monthly_macro_f1_diff:+.3f} | {r.wilcoxon_p:.3f} | {r.months_selected_better}/12 | "
                 f"{r.pooled_macro_f1_diff:+.3f} | [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.mcnemar_p:.3f} |")
    m = monthly.round(3)
    L += ["", "## 6. Macro-F1 per month", "", "| Month | " + " | ".join(short(c) for c in m.columns) + " |",
          "|---" * (len(m.columns) + 1) + "|"]
    L += [f"| {i} | " + " | ".join(f"{v:.3f}" for v in row) + " |" for i, row in m.iterrows()]
    L += ["", "## 7. How the evaluation protocol evolved (transparency)", "",
          "* **v1** (`model_selection_report_v1.md`): all models incl. rules ranked on 6 expanding folds; picked *seasonal naive*, "
          "the worst model on Jul-Dec 2024. Flaws: seasonal naive silently equals persistence where no previous year exists; "
          "tuned models were scored on their own tuning folds; rules give no probabilities.",
          "* **v2** (`model_selection_report_v2.md`): tuning folds separated from selection folds (Jan-Jun 2024), rules as benchmarks; "
          "picked *LightGBM*, which Jul-Dec 2024 then showed to be significantly worse than several models. Cause: three "
          "2-month folds cannot separate models whose differences (~0.01) are far smaller than the fold-to-fold noise (~0.05), "
          "and a fixed tie band of 0.005 ignored that uncertainty.",
          "* **v3** (this report): the use case defines 'best'; all 12 months of 2024 are evaluated by rolling origin; "
          "ties are decided by a significance test, not a fixed margin. Trade-off, stated openly: with only 24 months of data "
          "there is no untouched test period left after v3, and v3 was designed after v1/v2 results had been seen.",
          "", "## 8. Tuned hyperparameters", "", "```json", json.dumps(best_params, indent=2, default=str), "```"]
    (FINAL_REPORTS_DIR / "model_selection_report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    for d in (FINAL_MODELS_DIR, FINAL_REPORTS_DIR, FINAL_FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    bp_file = FINAL_MODELS_DIR / "best_params.json"
    best_params = json.loads(bp_file.read_text()) if bp_file.exists() else {}
    df, _, _ = load_split()                                  # all months, hold-out row excluded
    models = build_final_models({k: v for k, v in best_params.items() if not k.startswith("_")})

    print("== rolling-origin evaluation over 2024 ==")
    results, ev = rolling_evaluation(models, df)
    df_eval = df[ev]
    y = df_eval[TARGET].values
    keep = ("pred", "proba", "monthly", "deterministic")
    table = pd.DataFrame({n: {k: v for k, v in r.items() if k not in keep} for n, r in results.items()}).T.astype(float)
    table.insert(0, "group", [GROUPS[n] for n in table.index])
    monthly = pd.DataFrame({n: r["monthly"] for n, r in results.items()})

    selected, steps, top, tests = select(results)
    print("\n" + "\n".join(steps))
    sig = significance(results, selected, y)
    per_cat = per_technology(results, df_eval)

    print("\n== refit every model on all data (Feb-2023 .. Dec-2024) for deployment ==")
    X_all, y_all = df[FINAL_INPUT], df[TARGET].values
    files = {}
    for name, model in models.items():
        t0 = time.time()
        fitted = clone(model).fit(X_all, y_all)
        table.loc[name, "final_fit_seconds"] = time.time() - t0
        lat = []
        for _ in range(10):
            t1 = time.perf_counter()
            fitted.predict_proba(X_all.iloc[:1])
            lat.append(time.perf_counter() - t1)
        table.loc[name, "latency_ms"] = 1000 * float(np.median(lat))
        blob = pickle.dumps(fitted)
        table.loc[name, "size_mb"] = len(blob) / 1e6
        (FINAL_MODELS_DIR / f"{SLUGS[name]}.pkl").write_bytes(blob)
        files[name] = f"final/{SLUGS[name]}.pkl"

    table.to_csv(FINAL_REPORTS_DIR / "final_metrics.csv", float_format="%.4f")
    monthly.to_csv(FINAL_REPORTS_DIR / "monthly_macro_f1.csv", float_format="%.4f")
    sig.to_csv(FINAL_REPORTS_DIR / "significance.csv", index=False, float_format="%.4f")
    per_cat.to_csv(FINAL_REPORTS_DIR / "per_technology_macro_f1.csv", float_format="%.4f")
    for old in ("per_month_macro_f1.csv", "regime_change_accuracy.csv"):      # v2 outputs, superseded
        (FINAL_REPORTS_DIR / old).unlink(missing_ok=True)
    (FINAL_FIG_DIR / "f1_cv_vs_test.png").unlink(missing_ok=True)

    fig_scorecard(table, selected, top)
    fig_significance(sig, selected)
    fig_regime_monthly(table, monthly, selected)
    fig_confusion_and_tech(results, per_cat, y, selected)
    write_report(table, steps, selected, top, sig, monthly, best_params, int(ev.sum()))

    app_models = list(dict.fromkeys([selected, "CatBoost", "LightGBM", "Hybrid stack (Markov+LightGBM+RF)",
                                     "Soft Voting (RF+XGB)", "Markov chain"]))
    meta = {
        "task": "Final assignment: next-month capacity-factor tier (Low/Moderate/High) of Sri Lankan NCRE plants",
        "protocol": "v3 rolling-origin over 2024 (12 months); tuning on Jul-Dec 2023 only",
        "classes": CLASS_NAMES, "input_columns": FINAL_INPUT,
        "eval_months": [MONTH_LABEL[m] for m in EVAL_MONTHS], "eval_rows": int(ev.sum()),
        "deployment_training_period": "2023-02 .. 2024-12 (hold-out row excluded)",
        "selected_model": selected, "selection_steps": steps, "top_group": top, "wilcoxon_vs_best": tests,
        "model_files": files, "groups": GROUPS, "app_models": {n: files[n] for n in app_models},
        "metrics": table.round(4).to_dict("index"),
        "monthly_macro_f1": monthly.round(4).to_dict("index"),
        "significance_vs_selected": sig.round(4).to_dict("records"),
        "best_params": best_params,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (FINAL_MODELS_DIR / "final_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nSELECTED: {selected}")


if __name__ == "__main__":
    main()
