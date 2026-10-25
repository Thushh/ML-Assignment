"""Smoke tests: feature helper, saved model, and Flask routes.  Run:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from src.config import FEATURES, PROCESSED_DIR  # noqa: E402
from src.features import build_single_row, capacity_factor, cf_to_class  # noqa: E402


def test_capacity_factor_and_classes():
    # 1 MW for a 31-day month at full output = 0.744 GWh -> CF 1.0
    assert capacity_factor(0.744, 1.0, 2024, 1) == pytest.approx(1.0)
    assert [cf_to_class(v) for v in (0.05, 0.2, 0.6)] == [0, 1, 2]


def test_single_row_matches_training_columns():
    row = build_single_row("Solar", 1.0, 2025, 1, 0.12, 0.1, None, None, 600, 18)
    assert list(row.columns) == FEATURES + ["plant_id"]      # plant_id is used by the final-assignment CatBoost
    assert row.plant_id.iloc[0] == "unknown"
    assert np.isnan(row.cf_lag3.iloc[0]) and row.season.iloc[0] == "NE Monsoon"
    # CF > 100 % is invalid -> NaN (same rule as training)
    assert np.isnan(build_single_row("Solar", 1.0, 2025, 1, 5.0).cf_lag1.iloc[0])


def test_app_routes():
    from app import app
    c = app.test_client()
    assert c.get("/").status_code == 200
    hold = json.loads((PROCESSED_DIR / "holdout_sample.json").read_text())
    filled = c.get(f"/api/plant?plant={hold['plant_id']}&year={hold['year']}&month={hold['month']}").json
    res = c.post("/api/predict", json={**filled, "year": hold["year"], "month": hold["month"]}).json
    assert res["class"] in ("Low", "Moderate", "High")
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    bad = c.post("/api/predict", json={"category": "Solar", "capacity_mw": -2, "year": 2024, "month": 5})
    assert bad.status_code == 400


def test_explainers_are_faithful_to_the_saved_model():
    import pickle
    from src import explain_ensemble as E
    from src.config import MODELS_DIR
    model = pickle.load(open(MODELS_DIR / "model.pkl", "rb"))
    _, train, test = E.load_split()
    rows = test[FEATURES].sample(2, random_state=0).reset_index(drop=True)
    # category/season survive the numeric encoding SHAP and LIME need
    assert E.to_frame(E.to_numeric(rows)).fillna(-1).equals(rows.fillna(-1).astype(E.to_frame(E.to_numeric(rows)).dtypes))
    # SHAP: base value + contributions == the pipeline's probability, for every class
    sv = E.shap_explainer(model, train)(E.to_numeric(rows), max_evals=27)
    assert np.allclose(sv.values.sum(1) + sv.base_values, model.predict_proba(rows), atol=1e-6)
