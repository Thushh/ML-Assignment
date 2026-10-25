"""Project-wide constants: paths, class definition, feature lists, plot style."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

PLANT_FILE = RAW_DIR / "ceb_ncre_plant_monthly.csv"
SOURCE_FILE = RAW_DIR / "ceb_monthly_generation_by_source.csv"
SUMMARY_FILE = RAW_DIR / "ceb_system_summary_2023_2025.csv"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Target: monthly capacity-factor (CF) tier of a renewable (NCRE) plant.
#   CF = generation / (installed MW x hours in month)
#   Low      : CF < 15 %   -> plant is idle, in drought or under maintenance
#   Moderate : 15 % - 40 % -> typical solar / wind / average mini-hydro month
#   High     : CF >= 40 %  -> monsoon-fed hydro, firm dendro / waste plants
# ---------------------------------------------------------------------------
CF_LOW_MAX = 0.15
CF_HIGH_MIN = 0.40
CLASS_NAMES = ["Low", "Moderate", "High"]  # label 0, 1, 2

# Sri Lankan monsoon calendar (Department of Meteorology)
SEASON_BY_MONTH = {
    12: "NE Monsoon", 1: "NE Monsoon", 2: "NE Monsoon",
    3: "First Inter-monsoon", 4: "First Inter-monsoon",
    5: "SW Monsoon", 6: "SW Monsoon", 7: "SW Monsoon", 8: "SW Monsoon", 9: "SW Monsoon",
    10: "Second Inter-monsoon", 11: "Second Inter-monsoon",
}

CATEGORIES = ["Mini Hydro", "Solar", "Wind", "Dendro", "Biomass", "Municipal Waste"]

NUMERIC_LOG_FEATURES = ["installed_capacity_mw", "nat_hydro_lag1_gwh"]
NUMERIC_FEATURES = ["cf_lag1", "cf_lag2", "cf_lag3", "cf_lag12", "cf_roll3",
                    "cf_trend", "plant_mean_cf", "month_sin", "month_cos"]
CATEGORICAL_FEATURES = ["category", "season"]
FEATURES = NUMERIC_LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cf_class"

# Time-based split (t = 1 for Jan-2023 ... 24 for Dec-2024)
TRAIN_END_T = 18          # train: up to Jun-2024, test: Jul-Dec 2024
HOLDOUT_PLANT = "Kirkoswald"
HOLDOUT_T = 24            # Dec-2024 kept out of train AND test for inference demo

# Plot palette (validated categorical slots 1-3, light mode)
COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
          "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
CLASS_COLORS = [COLORS["orange"], COLORS["blue"], COLORS["aqua"]]

# ---------------------------------------------------------------------------
# FINAL ASSIGNMENT (Part 2) settings
# ---------------------------------------------------------------------------
FINAL_REPORTS_DIR = REPORTS_DIR / "final"
FINAL_FIG_DIR = FINAL_REPORTS_DIR / "figures"
FINAL_MODELS_DIR = MODELS_DIR / "final"

PLANT_ID = "plant_id"          # high-cardinality categorical (331 plants) - CatBoost/LightGBM can use it natively

# Expanding-window time-series CV: validate on 2-month blocks starting at these t values.
# Six folds cover Jul-2023 .. Jun-2024, i.e. a full monsoon cycle, before the Jul-Dec 2024 test.
FINAL_CV_STARTS = (7, 9, 11, 13, 15, 17)

# Folds 1-3 (Jul-Dec 2023) are used ONLY for tuning CatBoost / LightGBM;
# folds 4-6 (Jan-Jun 2024) are used ONLY for selecting the model -> no tuning bias in the selection score.
TUNE_CV_STARTS = (7, 9, 11)
SELECT_CV_STARTS = (13, 15, 17)

# Model-selection rule - protocol v3 (rolling-origin evaluation, use-case scorecard)
# Each month of 2024 (t = 13..24) is predicted by models trained on ALL earlier months.
EVAL_MONTHS = tuple(range(13, 25))
SIGNIFICANCE_LEVEL = 0.05      # top group = candidates NOT significantly worse than the best (Wilcoxon, monthly macro-F1)
SELECTION_TIE_BAND = 0.005     # (protocol v2 only, kept for the record)
N_BOOTSTRAP = 2000             # resamples for confidence intervals on the test set

FINAL_INPUT = FEATURES + [PLANT_ID]
