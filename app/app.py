"""Flask front end: loads the trained NGBoost model and returns a prediction, its uncertainty AND its explanation.

Run from the repository root:
    python app/app.py          then open http://127.0.0.1:5000
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import shap
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import (CATEGORIES, CLASS_NAMES, MODELS_DIR, PROCESSED_DIR, build_single_row,  # noqa: E402
                            capacity_factor, cf_to_class, class_gwh_band, hours_in_month, original_feature,
                            shift_month)

app = Flask(__name__)

META = json.loads((MODELS_DIR / "metadata.json").read_text())
MODEL_NAME = META["main_model"]
with open(MODELS_DIR / META["model_files"][MODEL_NAME], "rb") as fh:
    MODEL = pickle.load(fh)          # loaded once at start-up
EXPLAINER = shap.TreeExplainer(MODEL.ngb_, model_output=0)     # exact SHAP for the predicted mean capacity factor
COLUMNS = MODEL.prep_.get_feature_names_out()
LOOKUP = json.loads((PROCESSED_DIR / "plant_lookup.json").read_text())

LABELS = {"installed_capacity_mw": "Capacity", "nat_hydro_lag1_gwh": "National hydro last month",
          "cf_lag1": "Output last month", "cf_lag2": "Output 2 months ago", "cf_lag3": "Output 3 months ago",
          "cf_lag12": "Same month last year", "cf_roll3": "3-month average", "cf_trend": "Trend",
          "plant_mean_cf": "Plant's long-run average", "month_sin": "Time of year", "month_cos": "Time of year",
          "category": "Technology", "season": "Season"}


def explain(row, top=6):
    """Exact SHAP contributions of each input to the expected capacity factor (percentage points)."""
    values = EXPLAINER.shap_values(MODEL.prep_.transform(row))[0]
    per_feature = {}
    for col, v in zip(COLUMNS, values):                 # one-hot levels / missing flags -> original input
        label = LABELS.get(original_feature(col), original_feature(col))
        per_feature[label] = per_feature.get(label, 0.0) + float(v)
    order = sorted(per_feature, key=lambda f: -abs(per_feature[f]))[:top]
    return [{"feature": f, "contribution": round(per_feature[f] * 100, 1)} for f in order]


def _num(v):
    return None if v is None or str(v).strip() == "" else float(v)


def run_prediction(form):
    category = form.get("category")
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    capacity = _num(form.get("capacity_mw"))
    if not capacity or capacity <= 0:
        raise ValueError("Installed capacity must be a positive number (MW).")
    year, month = int(form.get("year")), int(form.get("month"))
    if not 1 <= month <= 12:
        raise ValueError("Month must be 1-12.")
    for key in ("gen_lag1_gwh", "gen_lag2_gwh", "gen_lag3_gwh", "gen_lag12_gwh", "nat_hydro_lag1_gwh"):
        v = _num(form.get(key))
        if v is not None and v < 0:
            raise ValueError("Generation values cannot be negative.")

    row = build_single_row(category, capacity, year, month, _num(form.get("gen_lag1_gwh")), _num(form.get("gen_lag2_gwh")),
                           _num(form.get("gen_lag3_gwh")), _num(form.get("gen_lag12_gwh")),
                           _num(form.get("nat_hydro_lag1_gwh")), _num(form.get("plant_mean_cf_pct")))
    proba = MODEL.predict_proba(row)[0]
    k = int(proba.argmax())
    mu, _ = MODEL.predict_dist(row)
    lo, hi = MODEL.predict_interval(row, 0.8)
    max_gwh = capacity * hours_in_month(year, month) / 1000.0
    return {"model": MODEL_NAME, "class": CLASS_NAMES[k], "class_index": k,
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, proba)},
            "expected_generation": class_gwh_band(k, capacity, year, month),
            "expected_cf_pct": round(float(np.clip(mu[0], 0, 1)) * 100, 1),
            "interval_cf_pct": [round(float(lo[0]) * 100, 1), round(float(hi[0]) * 100, 1)],
            "interval_gwh": [round(float(lo[0]) * max_gwh, 3), round(float(hi[0]) * max_gwh, 3)],
            "explanation": explain(row)}


@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORIES, plants=sorted(LOOKUP["plants"]), model_name=MODEL_NAME)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(run_prediction(data))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/plant")
def api_plant():
    """Autofill: a known plant's inputs for a target month, from CEB history."""
    pid, year, month = request.args.get("plant"), int(request.args["year"]), int(request.args["month"])
    plant = LOOKUP["plants"].get(pid)
    if plant is None:
        return jsonify({"error": "unknown plant"}), 404
    hist, cap = plant["history"], plant["capacity_mw"]

    def gen(k):
        y, m = shift_month(year, month, k)
        return hist.get(f"{y}-{m:02d}")

    target = f"{year}-{month:02d}"
    past_cf = [c for c in (capacity_factor(v, cap, int(key[:4]), int(key[5:])) for key, v in hist.items() if key < target) if c <= 1.0]
    y1, m1 = shift_month(year, month, 1)
    actual = hist.get(target)
    return jsonify({
        "category": plant["category"], "capacity_mw": cap,
        "gen_lag1_gwh": gen(1), "gen_lag2_gwh": gen(2), "gen_lag3_gwh": gen(3), "gen_lag12_gwh": gen(12),
        "nat_hydro_lag1_gwh": LOOKUP["national_major_hydro_gwh"].get(f"{y1}-{m1:02d}"),
        "plant_mean_cf_pct": round(float(np.mean(past_cf)) * 100, 1) if past_cf else None,
        "actual_gwh": actual,
        "actual_class": None if actual is None else CLASS_NAMES[cf_to_class(capacity_factor(actual, cap, year, month))],
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
