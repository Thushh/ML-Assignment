"""Step 1 - data collection, cleaning, feature engineering, encoding and normalisation.

Run:  python -m src.preprocess

Reads the raw CEB files in data/raw/ and writes:
    data/processed/model_dataset.csv   one row per plant per month: 13 features + target
    data/processed/holdout_sample.json one plant-month kept out of everything (inference demo)
    data/processed/plant_lookup.json   each plant's history (used by the web app to autofill)
    data/processed/cleaning_log.json   every cleaning decision with counts
    outputs/data_*.png                 data-analysis figures

This file also holds the project-wide settings and helpers that the other scripts and
the web app import, so a feature is computed the same way in training and in the app.
"""
import calendar
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ngboost import NGBRegressor  # noqa: E402
from ngboost.distns import Normal  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402

# =============================================================================== settings
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR, PROCESSED_DIR = ROOT / "data" / "raw", ROOT / "data" / "processed"
MODELS_DIR, OUTPUTS_DIR = ROOT / "models", ROOT / "outputs"
PLANT_FILE = RAW_DIR / "ceb_ncre_plant_monthly.csv"
SOURCE_FILE = RAW_DIR / "ceb_monthly_generation_by_source.csv"
RANDOM_STATE = 42

# Target: capacity-factor (CF) tier.  CF = GWh * 1000 / (MW * hours in the month)
CF_LOW_MAX, CF_HIGH_MIN = 0.15, 0.40
CLASS_NAMES = ["Low", "Moderate", "High"]          # labels 0, 1, 2
TARGET = "cf_class"

SEASON_BY_MONTH = {12: "NE Monsoon", 1: "NE Monsoon", 2: "NE Monsoon",
                   3: "First Inter-monsoon", 4: "First Inter-monsoon",
                   5: "SW Monsoon", 6: "SW Monsoon", 7: "SW Monsoon", 8: "SW Monsoon", 9: "SW Monsoon",
                   10: "Second Inter-monsoon", 11: "Second Inter-monsoon"}
CATEGORIES = ["Mini Hydro", "Solar", "Wind", "Dendro", "Biomass", "Municipal Waste"]

LOG_FEATURES = ["installed_capacity_mw", "nat_hydro_lag1_gwh"]          # right-skewed -> log
NUMERIC_FEATURES = ["cf_lag1", "cf_lag2", "cf_lag3", "cf_lag12", "cf_roll3", "cf_trend",
                    "plant_mean_cf", "month_sin", "month_cos"]
CATEGORICAL_FEATURES = ["category", "season"]
FEATURES = LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Time-based split (t = 1 for Jan-2023 ... 24 for Dec-2024)
TRAIN_END_T = 15   # train:      Feb-2023 .. Mar-2024
VAL_END_T = 18     # validation: Apr-2024 .. Jun-2024  (hyperparameter tuning)
                   # test:       Jul-2024 .. Dec-2024  (final evaluation, never used for tuning)
HOLDOUT_PLANT, HOLDOUT_T = "Kirkoswald", 24        # Dec-2024, kept out of train, validation AND test

COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "grey": "#a3a29c", "ink": "#1a2233"}
CLASS_COLORS = [COLORS["orange"], COLORS["blue"], COLORS["aqua"]]
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})


def save_fig(fig, name):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / name, dpi=150)
    plt.close(fig)


# =============================================================================== helpers
MONTH_ABBR = {calendar.month_abbr[i]: i for i in range(1, 13)}


def hours_in_month(year, month):
    return calendar.monthrange(int(year), int(month))[1] * 24


def capacity_factor(gwh, capacity_mw, year, month):
    if gwh is None or capacity_mw is None or capacity_mw <= 0:
        return np.nan
    return float(gwh) * 1000.0 / (float(capacity_mw) * hours_in_month(year, month))


def cf_to_class(cf):
    return 0 if cf < CF_LOW_MAX else 1 if cf < CF_HIGH_MIN else 2


def shift_month(year, month, k):
    """(year, month) k months earlier."""
    idx = int(year) * 12 + (int(month) - 1) - k
    return idx // 12, idx % 12 + 1


def month_cyclical(month):
    """Month on a circle, so December and January are neighbours."""
    angle = 2 * math.pi * (month - 1) / 12
    return math.sin(angle), math.cos(angle)


def build_single_row(category, capacity_mw, year, month, gen_lag1_gwh=None, gen_lag2_gwh=None,
                     gen_lag3_gwh=None, gen_lag12_gwh=None, nat_hydro_lag1_gwh=None, plant_mean_cf_pct=None):
    """User-friendly inputs (generation in GWh) -> one model-ready row with exactly FEATURES.

    Uses the same rules as training (CF > 100 % = invalid -> missing), so the web app and
    the training data are prepared identically.
    """
    year, month = int(year), int(month)

    def cf_k(gwh, k):
        if gwh in (None, ""):
            return np.nan
        y, m = shift_month(year, month, k)
        cf = capacity_factor(float(gwh), capacity_mw, y, m)
        return np.nan if cf > 1.0 else cf

    lag1, lag2, lag3, lag12 = cf_k(gen_lag1_gwh, 1), cf_k(gen_lag2_gwh, 2), cf_k(gen_lag3_gwh, 3), cf_k(gen_lag12_gwh, 12)
    lags = [lag1, lag2, lag3]
    sin_m, cos_m = month_cyclical(month)
    row = {
        "installed_capacity_mw": float(capacity_mw),
        "nat_hydro_lag1_gwh": np.nan if nat_hydro_lag1_gwh in (None, "") else float(nat_hydro_lag1_gwh),
        "cf_lag1": lag1, "cf_lag2": lag2, "cf_lag3": lag3, "cf_lag12": lag12,
        "cf_roll3": np.nan if np.all(np.isnan(lags)) else float(np.nanmean(lags)),
        "cf_trend": lag1 - lag2 if not (np.isnan(lag1) or np.isnan(lag2)) else np.nan,
        "plant_mean_cf": np.nan if plant_mean_cf_pct in (None, "") else float(plant_mean_cf_pct) / 100.0,
        "month_sin": sin_m, "month_cos": cos_m,
        "category": category, "season": SEASON_BY_MONTH[month],
    }
    return pd.DataFrame([row])[FEATURES]


def class_gwh_band(class_idx, capacity_mw, year, month):
    """Expected generation range (GWh) for a predicted class."""
    max_gwh = capacity_mw * hours_in_month(year, month) / 1000.0
    lo, hi = [(0, CF_LOW_MAX), (CF_LOW_MAX, CF_HIGH_MIN), (CF_HIGH_MIN, 1.0)][class_idx]
    return f"{lo * max_gwh:.3f} - {hi * max_gwh:.3f} GWh"


def make_preprocessor():
    """Encoding + normalisation used inside every model's pipeline except the EBM.
    Fitted on training data only.

    numeric:      median imputation (+ missing-value flag) -> standard scaling
    skewed:       median imputation (+ flag) -> log(1 + x) -> standard scaling
    categorical:  one-hot encoding
    The EBM does not need this: it handles missing values and text categories natively.
    """
    log_num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                    ("scale", StandardScaler())])
    return ColumnTransformer([("log_num", log_num, LOG_FEATURES), ("num", num, NUMERIC_FEATURES),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)])


class NGBoostTiers(BaseEstimator, ClassifierMixin):
    """Main model: NGBoost distributional regression on the capacity factor, used as a 3-tier classifier.

    fit() learns  CF ~ Normal(mean(x), sd(x)): every boosting round improves BOTH the mean and the
    spread, following the natural gradient of the log-likelihood. Tier probabilities come from it:
        P(Low) = P(CF < 0.15),   P(Moderate) = P(0.15 <= CF < 0.40),   P(High) = P(CF >= 0.40).
    (Defined here, in the shared module, so the saved model can be loaded by the web app.)
    """

    def __init__(self, n_estimators=500, learning_rate=0.03, max_depth=4, minibatch_frac=1.0):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_depth, self.minibatch_frac = max_depth, minibatch_frac

    def fit(self, X, cf):
        self.prep_ = make_preprocessor().fit(X)            # imputation, log, scaling, one-hot
        self.ngb_ = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=self.max_depth),
                                 n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                 minibatch_frac=self.minibatch_frac, verbose=False,
                                 random_state=RANDOM_STATE).fit(self.prep_.transform(X), np.asarray(cf))
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_dist(self, X):
        """Mean and standard deviation of next month's capacity factor."""
        p = self.ngb_.pred_dist(self.prep_.transform(X)).params
        return p["loc"], p["scale"]

    def predict_proba(self, X):
        mu, sd = self.predict_dist(X)
        low, below_high = norm.cdf(CF_LOW_MAX, mu, sd), norm.cdf(CF_HIGH_MIN, mu, sd)
        return np.c_[low, below_high - low, 1 - below_high]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def predict_interval(self, X, level=0.8):
        """Central prediction interval of the capacity factor, clipped to the physical range 0-1."""
        mu, sd = self.predict_dist(X)
        z = norm.ppf(0.5 + level / 2)
        return np.clip(mu - z * sd, 0, 1), np.clip(mu + z * sd, 0, 1)


def original_feature(column):
    """Map an encoded column name (e.g. 'cat__category_Solar', 'num__missingindicator_cf_lag12')
    back to the original feature it came from, so explanations can be shown per feature."""
    name = column.split("__", 1)[-1].replace("missingindicator_", "")
    return next((f for f in CATEGORICAL_FEATURES if name.startswith(f + "_")), name)


def load_split():
    """Train / validation / test frames, split by time; the hold-out row is excluded."""
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv")
    df = df[~df.is_holdout].sort_values(["t", "plant_id"])
    return df[df.t <= TRAIN_END_T], df[(df.t > TRAIN_END_T) & (df.t <= VAL_END_T)], df[df.t > VAL_END_T]


# =============================================================================== cleaning
# The 2023 data book truncates or renames some plants; map them to the 2024 names.
MANUAL_ALIASES = {"Dehiattakandiya DPP": "Dehiattakandiya", "Energy Waste (Munic-": "to Energy Waste"}


def build_alias_map(df):
    """Map a truncated 2023 name to the single 2024 name (same technology) that starts with it."""
    names23, names24 = set(df.loc[df.Year == 2023, "Plant_Name"]), set(df.loc[df.Year == 2024, "Plant_Name"])
    alias = dict(MANUAL_ALIASES)
    for old in sorted(names23 - names24):
        if old in alias:
            continue
        stem = old.rstrip("- ").lower()
        cat = df.loc[df.Plant_Name == old, "Category"].iloc[0]
        hits = [n for n in names24 - names23
                if n.lower().startswith(stem) and df.loc[df.Plant_Name == n, "Category"].iloc[0] == cat]
        if len(hits) == 1:
            alias[old] = hits[0]
    return alias


def load_and_clean(log):
    df = pd.read_csv(PLANT_FILE)
    log["raw_rows"], log["raw_null_values"] = len(df), int(df.isna().sum().sum())

    agg = df.Plant_Name.eq("Other Renewables")                       # 1. a national total, not a plant
    log["dropped_aggregate_rows"] = int(agg.sum())
    df = df[~agg].copy()

    alias = build_alias_map(df)                                      # 2. one name per plant across years
    log["name_aliases_applied"], log["rows_renamed"] = alias, int(df.Plant_Name.isin(alias).sum())
    df["Plant_Name"] = df.Plant_Name.replace(alias)

    df["month"] = df.Month.map(MONTH_ABBR)                           # 3. same name, several units -> unique id
    df["unit"] = df.groupby(["Year", "month", "Plant_Name"]).cumcount()
    df["plant_id"] = np.where(df.unit == 0, df.Plant_Name, df.Plant_Name + " (unit " + (df.unit + 1).astype(str) + ")")

    cap = df.groupby("plant_id").Installed_Capacity_MW.agg(["min", "max"])   # 4. capacity differs between years
    log["capacity_conflicts_fixed"] = cap[cap["min"] != cap["max"]].to_dict("index")
    df["installed_capacity_mw"] = df.plant_id.map(cap["max"])

    df["t"] = (df.Year - 2023) * 12 + df.month                       # 5. capacity factor; CF > 100 % is impossible
    df["hours"] = [hours_in_month(y, m) for y, m in zip(df.Year, df.month)]
    df["cf"] = df.Generation_GWh * 1000 / (df.installed_capacity_mw * df.hours)
    invalid = df.cf > 1.0
    log["invalid_cf_set_to_missing"] = int(invalid.sum())
    df.loc[invalid, "cf"] = np.nan

    never = df.groupby("plant_id").Generation_GWh.transform("sum").eq(0)    # 6. plants that never generated
    log["never_operated_plants_dropped"] = sorted(df.loc[never, "plant_id"].unique().tolist())
    df = df[~never].copy()

    first_on = df[df.Generation_GWh > 0].groupby("plant_id").t.min()       # 7. zeros before commissioning
    pre = df.t < df.plant_id.map(first_on)
    log["pre_commissioning_rows_dropped"] = int(pre.sum())
    df = df[~pre].copy()

    df = df.rename(columns={"Category": "category", "Generation_GWh": "generation_gwh", "Year": "year"})
    log["clean_rows"], log["clean_plants"] = len(df), int(df.plant_id.nunique())
    return df[["plant_id", "Plant_Name", "category", "installed_capacity_mw", "year", "month", "t", "generation_gwh", "cf"]]


def national_hydro():
    src = pd.read_csv(SOURCE_FILE)
    hyd = src[(src.Source == "Major Hydro") & (src.Ownership == "CEB")].copy()
    hyd["t"] = (hyd.Year - 2023) * 12 + hyd.Month.map(MONTH_ABBR)
    return hyd[["t", "Generation_GWh"]].rename(columns={"Generation_GWh": "nat_hydro_gwh"})


# =============================================================================== features
def engineer_features(df, log):
    """Every feature uses only months BEFORE the month being predicted (no leakage)."""
    base = df[["plant_id", "t", "cf"]]
    for k in (1, 2, 3, 12):                                          # CF k months ago (join on t, not row order)
        df = df.merge(base.assign(t=base.t + k).rename(columns={"cf": f"cf_lag{k}"}), on=["plant_id", "t"], how="left")
    df["cf_roll3"] = df[["cf_lag1", "cf_lag2", "cf_lag3"]].mean(axis=1, skipna=True)
    df["cf_trend"] = df.cf_lag1 - df.cf_lag2
    df = df.sort_values(["plant_id", "t"])
    df["plant_mean_cf"] = df.groupby("plant_id").cf.transform(lambda s: s.shift(1).expanding().mean())
    hyd = national_hydro()
    df = df.merge(hyd.assign(t=hyd.t + 1).rename(columns={"nat_hydro_gwh": "nat_hydro_lag1_gwh"}), on="t", how="left")
    sc = df.month.apply(month_cyclical)
    df["month_sin"], df["month_cos"] = [s for s, _ in sc], [c for _, c in sc]
    df["season"] = df.month.map(SEASON_BY_MONTH)

    no_target = df.cf.isna()                                         # can't learn without an answer
    no_hist = df[["cf_lag1", "cf_lag2", "cf_lag3"]].isna().all(axis=1)   # a plant's first month
    log["rows_without_valid_target_dropped"] = int(no_target.sum())
    log["rows_without_history_dropped"] = int((no_hist & ~no_target).sum())
    df = df[~no_target & ~no_hist].copy()
    df[TARGET] = df.cf.apply(cf_to_class)
    log["missing_values_left_for_imputation"] = {c: int(v) for c, v in df[FEATURES].isna().sum().items() if v}
    log["modelling_rows"] = len(df)
    log["class_distribution"] = {CLASS_NAMES[k]: int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    return df


def build_lookup(clean):
    hyd = national_hydro()
    plants = {pid: {"category": g.category.iloc[0], "capacity_mw": float(g.installed_capacity_mw.iloc[0]),
                    "history": {f"{int(r.year)}-{int(r.month):02d}": round(float(r.generation_gwh), 3) for r in g.itertuples()}}
              for pid, g in clean.groupby("plant_id")}
    nat = {f"{2023 + (int(t) - 1) // 12}-{(int(t) - 1) % 12 + 1:02d}": float(v) for t, v in zip(hyd.t, hyd.nat_hydro_gwh)}
    return {"plants": plants, "national_major_hydro_gwh": nat}


# =============================================================================== data analysis figures
def analysis_figures(data):
    """Training-period analysis that motivates the model choice (saved to outputs/)."""
    train = data[(data.t <= TRAIN_END_T) & ~data.is_holdout]

    ct = pd.crosstab(train.month, train[TARGET], normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bottom = np.zeros(len(ct))
    for k in range(3):
        ax.bar([calendar.month_abbr[m] for m in ct.index], ct[k], bottom=bottom, color=CLASS_COLORS[k], label=CLASS_NAMES[k])
        bottom += ct[k].values
    ax.set_ylabel("Share of plant-months")
    ax.set_title("Output tier by calendar month (training period): the monsoon pattern")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_fig(fig, "data_class_by_month.png")

    prev = train.cf_lag1.apply(lambda v: cf_to_class(v) if pd.notna(v) else -1)
    m = pd.crosstab(prev[prev >= 0], train[TARGET][prev >= 0], normalize="index")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", color="white" if m.values[i, j] > 0.5 else COLORS["ink"])
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Tier this month")
    ax.set_ylabel("Tier last month")
    ax.set_title("Tier-to-tier transitions")
    save_fig(fig, "data_transitions.png")
    return {"share_keeping_last_tier": round(float((prev[prev >= 0] == train[TARGET][prev >= 0]).mean()), 3)}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log = {}
    clean = load_and_clean(log)
    data = engineer_features(clean, log)
    hold = (data.Plant_Name == HOLDOUT_PLANT) & (data.t == HOLDOUT_T)
    data["is_holdout"] = hold
    log["split"] = {"train": "Feb-2023 .. Mar-2024", "validation": "Apr-2024 .. Jun-2024", "test": "Jul-2024 .. Dec-2024",
                    "rows": {"train": int(((data.t <= TRAIN_END_T) & ~hold).sum()),
                             "validation": int(((data.t > TRAIN_END_T) & (data.t <= VAL_END_T) & ~hold).sum()),
                             "test": int(((data.t > VAL_END_T) & ~hold).sum())},
                    "holdout": f"{HOLDOUT_PLANT} 2024-12"}
    log["analysis"] = analysis_figures(data)

    data.to_csv(PROCESSED_DIR / "model_dataset.csv", index=False)
    h = data[hold].iloc[0]
    sample = {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if hasattr(v, "item") else v))
              for k, v in h[["plant_id", "category", "installed_capacity_mw", "year", "month", "generation_gwh", "cf", TARGET] + FEATURES].to_dict().items()}
    sample["true_class"] = CLASS_NAMES[sample[TARGET]]
    (PROCESSED_DIR / "holdout_sample.json").write_text(json.dumps(sample, indent=2))
    (PROCESSED_DIR / "plant_lookup.json").write_text(json.dumps(build_lookup(clean)))
    (PROCESSED_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "name_aliases_applied"}, indent=2, default=str))


if __name__ == "__main__":
    main()
