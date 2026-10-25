"""Single-sample inference with the exported pickle model(s).

Run:
    python -m src.predict                      # predicts the left-out sample (Kirkoswald, Dec-2024)
    python -m src.predict --category "Mini Hydro" --capacity 4 --year 2024 --month 12 \
        --lag1 2.56 --lag2 1.122 --lag3 1.013 --lag12 2.978 --hydro 499.8 --mean-cf 30
"""
import argparse
import json
import pickle

import pandas as pd

from .config import CLASS_NAMES, FEATURES, MODELS_DIR, PROCESSED_DIR
from .features import build_single_row, class_gwh_band


def load_model(file_name: str = "model.pkl"):
    with open(MODELS_DIR / file_name, "rb") as fh:
        return pickle.load(fh)


def predict_row(model, row: pd.DataFrame) -> dict:
    proba = model.predict_proba(row)[0]
    idx = int(proba.argmax())
    return {"class_index": idx, "class": CLASS_NAMES[idx],
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, proba)}}


def predict_holdout():
    hold = json.loads((PROCESSED_DIR / "holdout_sample.json").read_text())
    row = pd.DataFrame([{f: hold[f] for f in FEATURES}]).astype({"category": str, "season": str})
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    print(f"Left-out sample: {hold['plant_id']} ({hold['category']}, {hold['installed_capacity_mw']} MW), "
          f"{hold['year']}-{hold['month']:02d}")
    print(f"Actual generation {hold['generation_gwh']} GWh -> CF {hold['cf']:.1%} -> TRUE class: {hold['true_class']}\n")
    for name, fname in meta["model_files"].items():
        res = predict_row(load_model(fname), row)
        band = class_gwh_band(res["class_index"], hold["installed_capacity_mw"], hold["year"], hold["month"])
        print(f"{name:30s} -> {res['class']:9s} {res['probabilities']}  (expected {band})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category")
    ap.add_argument("--capacity", type=float)
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    ap.add_argument("--lag1", type=float, help="generation last month (GWh)")
    ap.add_argument("--lag2", type=float, help="generation 2 months ago (GWh)")
    ap.add_argument("--lag3", type=float, help="generation 3 months ago (GWh)")
    ap.add_argument("--lag12", type=float, help="generation same month last year (GWh)")
    ap.add_argument("--hydro", type=float, help="national CEB major-hydro generation last month (GWh)")
    ap.add_argument("--mean-cf", type=float, help="plant's historical average capacity factor (%%)")
    ap.add_argument("--model", default="model.pkl")
    a = ap.parse_args()
    if a.category is None:
        predict_holdout()
        return
    row = build_single_row(a.category, a.capacity, a.year, a.month, a.lag1, a.lag2, a.lag3,
                           a.lag12, a.hydro, a.mean_cf)
    res = predict_row(load_model(a.model), row)
    res["expected_generation"] = class_gwh_band(res["class_index"], a.capacity, a.year, a.month)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
