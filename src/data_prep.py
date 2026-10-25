"""Data collection -> cleaning -> feature engineering for the CEB NCRE plant dataset.

Run:  python -m src.data_prep
Outputs (data/processed/):
    model_dataset.csv   one row per plant-month with features + target
    holdout_sample.json the single left-out row used for inference demo
    plant_lookup.json   per-plant generation history (front-end autofill)
    cleaning_log.json   every cleaning decision with row counts
"""
import json

import numpy as np
import pandas as pd

from .config import (CF_HIGH_MIN, CF_LOW_MAX, CLASS_NAMES, FEATURES, HOLDOUT_PLANT, HOLDOUT_T,
                     PLANT_FILE, PROCESSED_DIR, SEASON_BY_MONTH, SOURCE_FILE, TARGET)
from .features import MONTH_ABBR, cf_to_class, hours_in_month, month_cyclical

# 2023 data book truncates / renames some plants; map them to the 2024 names.
MANUAL_ALIASES = {
    "Dehiattakandiya DPP": "Dehiattakandiya",
    "Energy Waste (Munic-": "to Energy Waste",
}


def build_alias_map(df: pd.DataFrame) -> dict:
    """Auto-map truncated 2023 names to the unique 2024 name that starts with them."""
    names23 = set(df.loc[df.Year == 2023, "Plant_Name"])
    names24 = set(df.loc[df.Year == 2024, "Plant_Name"])
    new24 = names24 - names23
    alias = dict(MANUAL_ALIASES)
    for old in sorted(names23 - names24):
        if old in alias:
            continue
        stem = old.rstrip("- ").lower()
        cat = df.loc[df.Plant_Name == old, "Category"].iloc[0]
        hits = [n for n in new24 if n.lower().startswith(stem)
                and (df.loc[df.Plant_Name == n, "Category"].iloc[0] == cat)]
        if len(hits) == 1:
            alias[old] = hits[0]
    return alias


def load_and_clean(log: dict) -> pd.DataFrame:
    df = pd.read_csv(PLANT_FILE)
    log["raw_rows"] = len(df)
    log["raw_null_values"] = int(df.isna().sum().sum())

    # 1. aggregate line, not a plant
    agg = df.Plant_Name.eq("Other Renewables")
    log["dropped_aggregate_rows(Other Renewables)"] = int(agg.sum())
    df = df[~agg].copy()

    # 2. harmonise names across years
    alias = build_alias_map(df)
    log["name_aliases_applied"] = alias
    log["rows_renamed"] = int(df.Plant_Name.isin(alias).sum())
    df["Plant_Name"] = df.Plant_Name.replace(alias)

    # 3. same name used for several units -> unique plant_id (file order is stable)
    df["month"] = df.Month.map(MONTH_ABBR)
    df["unit"] = df.groupby(["Year", "month", "Plant_Name"]).cumcount()
    df["plant_id"] = np.where(df.unit == 0, df.Plant_Name,
                              df.Plant_Name + " (unit " + (df.unit + 1).astype(str) + ")")
    log["multi_unit_plants"] = sorted(df.loc[df.unit > 0, "Plant_Name"].unique().tolist())

    # 4. inconsistent capacity across years -> use the max reported nameplate
    cap = df.groupby("plant_id").Installed_Capacity_MW.agg(["min", "max"])
    log["capacity_conflicts_fixed"] = cap[cap["min"] != cap["max"]].to_dict("index")
    df["installed_capacity_mw"] = df.plant_id.map(cap["max"])

    # 5. capacity factor + null detection of physically impossible values (CF > 100 %)
    df["t"] = (df.Year - 2023) * 12 + df.month
    df["hours"] = [hours_in_month(y, m) for y, m in zip(df.Year, df.month)]
    df["cf"] = df.Generation_GWh * 1000 / (df.installed_capacity_mw * df.hours)
    invalid = df.cf > 1.0
    log["invalid_cf_gt_100pct_set_to_null"] = int(invalid.sum())
    df.loc[invalid, "cf"] = np.nan

    # 6. plants that never generated (24 x zero) carry no signal
    never = df.groupby("plant_id").Generation_GWh.transform("sum").eq(0)
    log["never_operated_plants_dropped"] = sorted(df.loc[never, "plant_id"].unique().tolist())
    df = df[~never].copy()

    # 7. zeros before commissioning are not outages
    first_on = df[df.Generation_GWh > 0].groupby("plant_id").t.min()
    pre = df.t < df.plant_id.map(first_on)
    log["pre_commissioning_rows_dropped"] = int(pre.sum())
    df = df[~pre].copy()

    df = df.rename(columns={"Category": "category", "Generation_GWh": "generation_gwh",
                            "Year": "year"})
    log["clean_rows"] = len(df)
    log["clean_plants"] = int(df.plant_id.nunique())
    return df[["plant_id", "Plant_Name", "category", "installed_capacity_mw", "year",
               "month", "t", "generation_gwh", "cf"]]


def national_hydro() -> pd.DataFrame:
    src = pd.read_csv(SOURCE_FILE)
    hyd = src[(src.Source == "Major Hydro") & (src.Ownership == "CEB")].copy()
    hyd["t"] = (hyd.Year - 2023) * 12 + hyd.Month.map(MONTH_ABBR)
    return hyd[["t", "Generation_GWh"]].rename(columns={"Generation_GWh": "nat_hydro_gwh"})


def engineer_features(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    base = df[["plant_id", "t", "cf"]]
    for k in (1, 2, 3, 12):
        lag = base.assign(t=base.t + k).rename(columns={"cf": f"cf_lag{k}"})
        df = df.merge(lag, on=["plant_id", "t"], how="left")
    df["cf_roll3"] = df[["cf_lag1", "cf_lag2", "cf_lag3"]].mean(axis=1, skipna=True)
    df["cf_trend"] = df.cf_lag1 - df.cf_lag2          # rising / falling output
    # plant's own long-run level: expanding mean of all PAST months (no leakage)
    df = df.sort_values(["plant_id", "t"])
    df["plant_mean_cf"] = df.groupby("plant_id").cf.transform(lambda s: s.shift(1).expanding().mean())

    hyd = national_hydro()
    df = df.merge(hyd.assign(t=hyd.t + 1).rename(columns={"nat_hydro_gwh": "nat_hydro_lag1_gwh"}),
                  on="t", how="left")

    sc = df.month.apply(month_cyclical)
    df["month_sin"] = [s for s, _ in sc]
    df["month_cos"] = [c for _, c in sc]
    df["season"] = df.month.map(SEASON_BY_MONTH)

    # target rows: valid CF and at least some history
    no_target = df.cf.isna()
    no_hist = df[["cf_lag1", "cf_lag2", "cf_lag3"]].isna().all(axis=1)
    log["rows_without_valid_target_dropped"] = int(no_target.sum())
    log["rows_without_history_dropped(first month of each plant)"] = int((no_hist & ~no_target).sum())
    df = df[~no_target & ~no_hist].copy()
    df[TARGET] = df.cf.apply(cf_to_class)

    log["missing_values_left_for_imputation"] = {c: int(v) for c, v in df[FEATURES].isna().sum().items() if v}
    log["modelling_rows"] = len(df)
    log["class_distribution"] = {CLASS_NAMES[k]: int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    log["class_thresholds"] = {"Low": f"CF < {CF_LOW_MAX}", "Moderate": f"{CF_LOW_MAX} <= CF < {CF_HIGH_MIN}",
                               "High": f"CF >= {CF_HIGH_MIN}"}
    return df


def build_lookup(clean: pd.DataFrame) -> dict:
    hyd = national_hydro()
    plants = {}
    for pid, g in clean.groupby("plant_id"):
        plants[pid] = {
            "category": g.category.iloc[0],
            "capacity_mw": float(g.installed_capacity_mw.iloc[0]),
            "history": {f"{int(r.year)}-{int(r.month):02d}": round(float(r.generation_gwh), 3)
                        for r in g.itertuples()},
        }
    nat = {f"{2023 + (int(t) - 1) // 12}-{(int(t) - 1) % 12 + 1:02d}": float(v)
           for t, v in zip(hyd.t, hyd.nat_hydro_gwh)}
    return {"plants": plants, "national_major_hydro_gwh": nat}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log = {}
    clean = load_and_clean(log)
    data = engineer_features(clean, log)

    hold_mask = (data.Plant_Name == HOLDOUT_PLANT) & (data.t == HOLDOUT_T)
    holdout = data[hold_mask].iloc[0]
    data["is_holdout"] = hold_mask
    log["holdout"] = f"{holdout.plant_id} {int(holdout.year)}-{int(holdout.month):02d}"

    data.to_csv(PROCESSED_DIR / "model_dataset.csv", index=False)
    hold = {k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in holdout[["plant_id", "category", "installed_capacity_mw", "year", "month",
                                 "generation_gwh", "cf", TARGET] + FEATURES].to_dict().items()}
    hold = {k: (v.item() if hasattr(v, "item") else v) for k, v in hold.items()}
    hold["true_class"] = CLASS_NAMES[hold[TARGET]]
    (PROCESSED_DIR / "holdout_sample.json").write_text(json.dumps(hold, indent=2))
    (PROCESSED_DIR / "plant_lookup.json").write_text(json.dumps(build_lookup(clean)))
    (PROCESSED_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "name_aliases_applied"}, indent=2, default=str))


if __name__ == "__main__":
    main()
