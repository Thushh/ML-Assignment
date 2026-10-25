"""Shared feature helpers used by data prep, training, CLI inference and the web app.

Keeping the single-sample feature builder here guarantees the front end feeds the
model exactly the same columns the model was trained on.
"""
import calendar
import math

import numpy as np
import pandas as pd

from .config import CF_HIGH_MIN, CF_LOW_MAX, CLASS_NAMES, FEATURES, SEASON_BY_MONTH

MONTH_ABBR = {calendar.month_abbr[i]: i for i in range(1, 13)}


def hours_in_month(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1] * 24


def capacity_factor(gwh, capacity_mw, year, month):
    """GWh -> capacity factor (0-1). Returns NaN when inputs are missing/invalid."""
    if gwh is None or capacity_mw is None or capacity_mw <= 0:
        return np.nan
    return float(gwh) * 1000.0 / (float(capacity_mw) * hours_in_month(year, month))


def cf_to_class(cf: float) -> int:
    if cf < CF_LOW_MAX:
        return 0
    if cf < CF_HIGH_MIN:
        return 1
    return 2


def shift_month(year: int, month: int, k: int):
    """Return (year, month) k months before the given month."""
    idx = year * 12 + (month - 1) - k
    return idx // 12, idx % 12 + 1


def month_cyclical(month: int):
    angle = 2 * math.pi * (month - 1) / 12
    return math.sin(angle), math.cos(angle)


def build_single_row(category, capacity_mw, year, month,
                     gen_lag1_gwh=None, gen_lag2_gwh=None, gen_lag3_gwh=None,
                     gen_lag12_gwh=None, nat_hydro_lag1_gwh=None,
                     plant_mean_cf_pct=None, plant_id=None) -> pd.DataFrame:
    """Turn user-friendly inputs (GWh history) into one model-ready feature row.

    Any history value left empty becomes NaN and is imputed inside the saved
    pipeline, exactly like missing lags during training.
    """
    year, month = int(year), int(month)

    def cf_k(gwh, k):
        if gwh in (None, ""):
            return np.nan
        y, m = shift_month(year, month, k)
        cf = capacity_factor(float(gwh), capacity_mw, y, m)
        return np.nan if cf > 1.0 else cf   # same null rule as training (CF > 100 % = invalid record)

    lag1, lag2, lag3 = cf_k(gen_lag1_gwh, 1), cf_k(gen_lag2_gwh, 2), cf_k(gen_lag3_gwh, 3)
    lag12 = cf_k(gen_lag12_gwh, 12)
    roll3 = np.nanmean([lag1, lag2, lag3]) if not np.all(np.isnan([lag1, lag2, lag3])) else np.nan
    trend = lag1 - lag2 if not (np.isnan(lag1) or np.isnan(lag2)) else np.nan
    mean_cf = np.nan if plant_mean_cf_pct in (None, "") else float(plant_mean_cf_pct) / 100.0
    sin_m, cos_m = month_cyclical(month)
    row = {
        "installed_capacity_mw": float(capacity_mw),
        "nat_hydro_lag1_gwh": np.nan if nat_hydro_lag1_gwh in (None, "") else float(nat_hydro_lag1_gwh),
        "cf_lag1": lag1, "cf_lag2": lag2, "cf_lag3": lag3, "cf_lag12": lag12, "cf_roll3": roll3,
        "cf_trend": trend, "plant_mean_cf": mean_cf,
        "month_sin": sin_m, "month_cos": cos_m,
        "category": category, "season": SEASON_BY_MONTH[month],
    }
    out = pd.DataFrame([row])[FEATURES]
    # Final-assignment models (CatBoost / LightGBM) can also use the plant's identity.
    # Activity models ignore this extra column. Unknown plants are handled natively.
    out["plant_id"] = plant_id if plant_id not in (None, "") else "unknown"
    return out


def class_gwh_band(class_idx: int, capacity_mw: float, year: int, month: int) -> str:
    """Human-readable expected generation range for a predicted class."""
    max_gwh = capacity_mw * hours_in_month(year, month) / 1000.0
    lo, hi = [(0, CF_LOW_MAX), (CF_LOW_MAX, CF_HIGH_MIN), (CF_HIGH_MIN, 1.0)][class_idx]
    return f"{lo * max_gwh:.3f} - {hi * max_gwh:.3f} GWh"


__all__ = ["CLASS_NAMES", "build_single_row", "capacity_factor", "cf_to_class",
           "class_gwh_band", "hours_in_month", "shift_month", "MONTH_ABBR"]
