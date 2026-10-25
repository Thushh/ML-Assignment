"""Flask front end: pickle loader + form -> prediction for the CEB plant output-tier model.

Run from the repository root:
    python app/app.py          then open http://127.0.0.1:5000
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CATEGORIES, CLASS_NAMES, FINAL_MODELS_DIR, MODELS_DIR, PROCESSED_DIR  # noqa: E402
from src.features import build_single_row, capacity_factor, cf_to_class, class_gwh_band, shift_month  # noqa: E402

app = Flask(__name__)

# Load ONLY the chosen model: the final assignment's selected model if it exists,
# otherwise the class activity's default model.
FINAL_META_FILE = FINAL_MODELS_DIR / "final_metadata.json"
if FINAL_META_FILE.exists():
    _meta = json.loads(FINAL_META_FILE.read_text())
    MODEL_NAME = _meta["selected_model"]
    MODEL_FILE = MODELS_DIR / _meta["model_files"][MODEL_NAME]
else:
    _meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    MODEL_NAME = _meta["default_model"]
    MODEL_FILE = MODELS_DIR / _meta["model_files"][MODEL_NAME]
with open(MODEL_FILE, "rb") as fh:
    MODEL = pickle.load(fh)
LOOKUP = json.loads((PROCESSED_DIR / "plant_lookup.json").read_text())


def _num(v):
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def run_prediction(form: dict) -> dict:
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

    row = build_single_row(category, capacity, year, month,
                           _num(form.get("gen_lag1_gwh")), _num(form.get("gen_lag2_gwh")),
                           _num(form.get("gen_lag3_gwh")), _num(form.get("gen_lag12_gwh")),
                           _num(form.get("nat_hydro_lag1_gwh")), _num(form.get("plant_mean_cf_pct")),
                           plant_id=form.get("plant_id"))
    proba = MODEL.predict_proba(row)[0]
    idx = int(proba.argmax())
    return {
        "model": MODEL_NAME, "class": CLASS_NAMES[idx], "class_index": idx,
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, proba)},
        "expected_generation": class_gwh_band(idx, capacity, year, month),
    }


@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORIES, plants=sorted(LOOKUP["plants"]),
                           model_name=MODEL_NAME)


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

    target_key = f"{year}-{month:02d}"
    past_cf = [capacity_factor(v, cap, int(k[:4]), int(k[5:])) for k, v in hist.items() if k < target_key]
    past_cf = [c for c in past_cf if c <= 1.0]
    y1, m1 = shift_month(year, month, 1)
    actual = hist.get(target_key)
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
