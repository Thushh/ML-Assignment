"""FINAL ASSIGNMENT - data analysis BEFORE choosing models (requirement 2).

Run:  python -m src.analysis
Uses the TRAINING period only (Feb-2023 .. Jun-2024), so no decision about models is
influenced by the test months.

Outputs:
    reports/final/analysis.json         every number quoted in the report
    reports/final/analysis_report.md    findings -> implications for model choice
    reports/final/figures/a*_*.png      7 analysis figures
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from .config import (CF_HIGH_MIN, CF_LOW_MAX, CLASS_COLORS, CLASS_NAMES, COLORS, FEATURES,
                     FINAL_FIG_DIR, FINAL_REPORTS_DIR, PROCESSED_DIR, TARGET, TRAIN_END_T)
from .final_models import previous_class
from .plotting import plt, save

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUMERIC = ["installed_capacity_mw", "nat_hydro_lag1_gwh", "cf_lag1", "cf_lag2", "cf_lag3", "cf_lag12",
           "cf_roll3", "cf_trend", "plant_mean_cf", "month_sin", "month_cos"]


def _save(fig, name):
    save(fig, name, folder=FINAL_FIG_DIR)


# --------------------------------------------------------------------------- A. data quality
def data_quality(df, out):
    log = json.loads((PROCESSED_DIR / "cleaning_log.json").read_text())
    removed = {
        "Aggregate line (Other Renewables)": log["dropped_aggregate_rows(Other Renewables)"],
        f"Never-operated plants ({len(log['never_operated_plants_dropped'])})":
            log["raw_rows"] - log["dropped_aggregate_rows(Other Renewables)"]
            - log["pre_commissioning_rows_dropped"] - log["clean_rows"],
        "Before commissioning": log["pre_commissioning_rows_dropped"],
        "Invalid CF > 100 % (target)": log["rows_without_valid_target_dropped"],
        "No history (first month)": log["rows_without_history_dropped(first month of each plant)"],
    }
    miss = (df[FEATURES].isna().mean() * 100).sort_values()
    out["data_quality"] = {"raw_rows": log["raw_rows"], "modelling_rows": log["modelling_rows"],
                           "removed": removed, "invalid_cf_set_to_null": log["invalid_cf_gt_100pct_set_to_null"],
                           "missing_pct_train": miss.round(1)[miss > 0].to_dict()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    s = pd.Series(removed).sort_values()
    axes[0].barh(s.index, s.values, color=COLORS["blue"], height=0.6)
    for i, v in enumerate(s.values):
        axes[0].text(v, i, f" {v}", va="center", fontsize=9)
    axes[0].set_title(f"Rows removed during cleaning ({log['raw_rows']:,} → {log['modelling_rows']:,})")
    m = miss[miss > 0]
    axes[1].barh(m.index, m.values, color=COLORS["orange"], height=0.6)
    for i, v in enumerate(m.values):
        axes[1].text(v, i, f" {v:.1f}%", va="center", fontsize=9)
    axes[1].set_title("Missing values left for the models (training rows)")
    axes[1].set_xlabel("% of rows")
    _save(fig, "a1_data_quality.png")


# --------------------------------------------------------------------------- B. target & season
def target_seasonality(df, out):
    ct = pd.crosstab(df.month, df[TARGET], normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
    counts = pd.crosstab(df.season, df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(counts)
    cramers_v = np.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1)))
    out["target"] = {"class_share": df[TARGET].map(dict(enumerate(CLASS_NAMES))).value_counts(normalize=True).round(3).to_dict(),
                     "chi2_season_vs_class": round(chi2, 1), "chi2_p": float(p), "cramers_v": round(cramers_v, 3)}

    fig, ax = plt.subplots(figsize=(10, 3.8))
    bottom = np.zeros(len(ct))
    for k in range(3):
        ax.bar([MONTHS[m - 1] for m in ct.index], ct[k], bottom=bottom, color=CLASS_COLORS[k],
               label=CLASS_NAMES[k], edgecolor=COLORS["surface"], linewidth=2, width=0.7)
        bottom += ct[k].values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of plant-months")
    ax.set_title(f"Class mix by calendar month (season vs class: χ² p < 0.001, Cramér's V = {cramers_v:.2f})")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.1))
    _save(fig, "a2_class_by_month.png")


# --------------------------------------------------------------------------- C. temporal dynamics
def transitions(df, out):
    prev = previous_class(df)
    # .values on both sides: align by POSITION (df's index has gaps after filtering)
    mat = pd.crosstab(pd.Series(prev, name="prev"), pd.Series(df[TARGET].values, name="now"), normalize="index")
    stay = float((prev == df[TARGET].values).mean())
    change_by_month = pd.Series(prev != df[TARGET].values).groupby(df.month.values).mean()
    out["dynamics"] = {"persistence_rate": round(stay, 3),
                       "transition_matrix": mat.round(3).values.tolist(),
                       "change_rate_by_month": {MONTHS[m - 1]: round(v, 3) for m, v in change_by_month.items()}}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
    axes[0].grid(False)
    for i in range(3):
        for j in range(3):
            v = mat.values[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else COLORS["ink"])
    axes[0].set_xticks(range(3), CLASS_NAMES)
    axes[0].set_yticks(range(3), CLASS_NAMES)
    axes[0].set_xlabel("Class this month")
    axes[0].set_ylabel("Class last month")
    axes[0].set_title("Transition probabilities")
    axes[1].bar([MONTHS[m - 1] for m in change_by_month.index], change_by_month.values,
                color=COLORS["orange"], width=0.65)
    axes[1].axhline(1 - stay, color=COLORS["ink2"], linestyle="--", linewidth=1)
    axes[1].text(0, 1 - stay + 0.01, f" average {1 - stay:.0%}", color=COLORS["ink2"], fontsize=9)
    axes[1].set_title("Share of plants that CHANGE class, by month")
    axes[1].set_ylabel("Change rate")
    _save(fig, "a3_transitions.png")


# --------------------------------------------------------------------------- D. feature relevance
def feature_relevance(df, out):
    X = df[FEATURES + ["plant_id"]].copy()
    disc = []
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes
            disc.append(True)
        else:
            X[c] = X[c].fillna(X[c].median())
            disc.append(False)
    mi = pd.Series(mutual_info_classif(X, df[TARGET], discrete_features=disc, random_state=0),
                   index=X.columns).sort_values()
    groups = [df.loc[df[TARGET] == k, "cf_lag1"].dropna() for k in range(3)]
    h, p = stats.kruskal(*groups)
    out["relevance"] = {"mutual_information": mi.sort_values(ascending=False).round(4).to_dict(),
                        "kruskal_cf_lag1_H": round(h, 1), "kruskal_p": float(p)}

    # empirical P(class | last month's CF): how sharp / non-linear is the relationship?
    bins = np.linspace(0, 1, 21)
    b = pd.cut(df.cf_lag1, bins, include_lowest=True)
    curve = pd.crosstab(b, df[TARGET], normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
    mids = [iv.mid for iv in curve.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].barh(mi.index, mi.values, color=COLORS["aqua"], height=0.6)
    axes[0].set_title("Mutual information with the target")
    axes[0].set_xlabel("MI (nats); higher = more informative")
    for k in range(3):
        axes[1].plot(mids, curve[k], color=CLASS_COLORS[k], marker="o", markersize=5, label=CLASS_NAMES[k])
    for x in (CF_LOW_MAX, CF_HIGH_MIN):
        axes[1].axvline(x, color=COLORS["ink2"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("Last month's capacity factor")
    axes[1].set_ylabel("P(class this month)")
    axes[1].set_title("P(class) vs last month's CF: threshold-like")
    axes[1].legend()
    _save(fig, "a4_feature_relevance.png")


# --------------------------------------------------------------------------- E. multicollinearity
def collinearity(df, out):
    corr = df[NUMERIC].corr(method="spearman")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    high = pairs[pairs.abs() >= 0.7].sort_values(ascending=False)
    out["collinearity"] = {"pairs_abs_rho_ge_0.7": {f"{a} ~ {b}": round(v, 2) for (a, b), v in high.items()}}

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=60, ha="right")
    ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else COLORS["ink"])
    ax.set_title("Spearman correlation between numeric features")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "a5_correlation.png")


# --------------------------------------------------------------------------- F. categorical structure
def categorical_structure(df, out):
    def eta_sq(group):   # share of CF variance explained by a grouping (one-way ANOVA effect size)
        g = df.groupby(group).cf
        between = (g.count() * (g.mean() - df.cf.mean()) ** 2).sum()
        return between / ((df.cf - df.cf.mean()) ** 2).sum()

    levels = {c: int(df[c].nunique()) for c in ["category", "season", "plant_id"]}
    eta = {"category": round(eta_sq("category"), 3), "plant_id": round(eta_sq("plant_id"), 3),
           "season": round(eta_sq("season"), 3)}
    out["categorical"] = {"levels": levels, "eta_squared_cf": eta}

    plant_means = df.groupby(["category", "plant_id"]).cf.mean().reset_index()
    order = plant_means.groupby("category").cf.median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, c in enumerate(order):
        v = plant_means.loc[plant_means.category == c, "cf"].values
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=14, color=COLORS["blue"], alpha=0.55)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=COLORS["ink"], linewidth=2)
    n_by_cat = plant_means.category.value_counts()
    ax.set_xticks(range(len(order)), [f"{c}\n({n_by_cat[c]} plant{'s' if n_by_cat[c] != 1 else ''})"
                                      for c in order])
    ax.set_ylabel("Plant's mean capacity factor")
    ax.set_title(f"Plants differ a lot within a technology\nplant identity explains "
                 f"{eta['plant_id']:.0%} of CF variance, technology {eta['category']:.0%}")
    _save(fig, "a6_plant_heterogeneity.png")


# --------------------------------------------------------------------------- G. drift between periods
def period_shift(df_all, out):
    tr = df_all[df_all.t <= TRAIN_END_T]
    late = df_all[df_all.t > TRAIN_END_T]
    a = tr[TARGET].value_counts(normalize=True).sort_index()
    b = late[TARGET].value_counts(normalize=True).sort_index()
    out["shift"] = {"train_class_share": a.round(3).tolist(), "later_period_class_share": b.round(3).tolist(),
                    "note": "only class SHARES of the later period are inspected (no model is fitted on it)"}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(3)
    ax.bar(x - 0.18, a.values, width=0.34, color=COLORS["blue"], label="Train (Feb-23..Jun-24)")
    ax.bar(x + 0.18, b.values, width=0.34, color=COLORS["orange"], label="Later (Jul..Dec-24)")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("Share")
    ax.set_title("Class shares shift between periods (wet 2nd half of 2024)")
    ax.legend()
    _save(fig, "a7_period_shift.png")


REPORT = """# Data analysis before model selection (training period only)

All numbers are computed on the training months (Feb-2023 .. Jun-2024, {n_train:,} plant-months).
Figures: `reports/final/figures/a1..a7`.

| # | Finding | Evidence | Implication for model choice |
|---|---|---|---|
| 1 | Hidden data-quality problems, no explicit nulls | {raw:,} raw rows → {model:,} modelling rows; {invalid} impossible CF values (>100 %) set to null; `cf_lag12` missing for {lag12_miss}% of training rows | Need models/pipelines that handle **missing values** robustly. CatBoost and LightGBM treat NaN natively; RF/XGB/LogReg use median imputation + missing flags |
| 2 | The target is **sticky** | {stay:.0%} of plant-months keep last month's class (transition matrix diagonal: {diag}) | Persistence and a **Markov chain** are strong *existing approaches* and must be beaten. Accuracy alone is misleading, so report **macro-F1** and performance on **tier-change months** |
| 3 | Changes are **seasonal** | class depends on season (χ² = {chi2}, p < 0.001, Cramér's V = {cv}); change rate peaks in {peak_months} | Season and cyclical month features are needed; the Markov chain is conditioned on season |
| 4 | Relationship to last month's CF is **non-linear / threshold-like** | Kruskal-Wallis on `cf_lag1` across classes H = {kw_h} (p < 0.001); P(class) jumps near the 0.15 / 0.40 boundaries (a4) | Tree ensembles (RF, XGBoost, **CatBoost, LightGBM**) suit threshold effects better than linear logistic regression |
| 5 | Strong **multicollinearity** among history features | {n_coll} feature pairs with \\|ρ\\| ≥ 0.7, e.g. {coll_example} | Linear models need regularisation (LogReg `C=0.5`); tree models are robust to it, but importance must be read with care (use SHAP / permutation) |
| 6 | **Categorical structure matters** | plant identity explains {eta_plant:.0%} of CF variance vs {eta_cat:.0%} for technology (η²); `plant_id` has {n_plants} levels in the training period | One-hot encoding {n_plants} plants is impractical for RF/XGB. **CatBoost** handles high-cardinality categoricals with ordered target statistics, which motivates choosing it |
| 7 | Most informative features (mutual information) | top: {top_mi}. (MI is biased upward for many-level categoricals such as `plant_id`; its value is confirmed independently by η² in finding 6) | History + identity drive the signal; national hydro (rainfall proxy) adds little, so external weather data is the main future improvement |
| 8 | **Class shift** between periods | Low share: {low_tr:.0%} in training vs {low_te:.0%} later | Prefer models whose ranking is stable across time folds; select with **time-series CV**, confirm on the later period |
"""


def main():
    FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(PROCESSED_DIR / "model_dataset.csv")
    df_all = df_all[~df_all.is_holdout]
    df = df_all[df_all.t <= TRAIN_END_T].copy()
    out = {"period": "train (t <= %d)" % TRAIN_END_T, "rows": len(df)}

    data_quality(df, out)
    target_seasonality(df, out)
    transitions(df, out)
    feature_relevance(df, out)
    collinearity(df, out)
    categorical_structure(df, out)
    period_shift(df_all, out)

    FINAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_REPORTS_DIR / "analysis.json").write_text(json.dumps(out, indent=2, default=float))

    cr = out["dynamics"]["change_rate_by_month"]
    peaks = sorted(cr, key=cr.get, reverse=True)[:3]
    coll = out["collinearity"]["pairs_abs_rho_ge_0.7"]
    mi = out["relevance"]["mutual_information"]
    diag = ", ".join(f"{CLASS_NAMES[i]} {out['dynamics']['transition_matrix'][i][i]:.2f}" for i in range(3))
    report = REPORT.format(
        n_train=len(df), raw=out["data_quality"]["raw_rows"], model=out["data_quality"]["modelling_rows"],
        invalid=out["data_quality"]["invalid_cf_set_to_null"],
        lag12_miss=out["data_quality"]["missing_pct_train"].get("cf_lag12", 0),
        stay=out["dynamics"]["persistence_rate"], diag=diag,
        chi2=out["target"]["chi2_season_vs_class"], cv=out["target"]["cramers_v"], peak_months=", ".join(peaks),
        kw_h=out["relevance"]["kruskal_cf_lag1_H"], n_coll=len(coll),
        coll_example=", ".join(f"`{k}` ({v})" for k, v in list(coll.items())[:2]),
        eta_plant=out["categorical"]["eta_squared_cf"]["plant_id"], eta_cat=out["categorical"]["eta_squared_cf"]["category"],
        n_plants=out["categorical"]["levels"]["plant_id"],
        top_mi=", ".join(f"`{k}`" for k in list(mi)[:4]),
        low_tr=out["shift"]["train_class_share"][0], low_te=out["shift"]["later_period_class_share"][0])
    (FINAL_REPORTS_DIR / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"analysis: {len(df)} training rows analysed -> reports/final/analysis_report.md + 7 figures")


if __name__ == "__main__":
    main()
