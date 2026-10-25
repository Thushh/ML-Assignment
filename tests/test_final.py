"""Tests for the final-assignment models.  Run:  python -m pytest -q"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURES, FINAL_INPUT, FINAL_MODELS_DIR, MODELS_DIR, PROCESSED_DIR, TARGET  # noqa: E402
from src.final_models import (MarkovChainModel, ModelFrame, PersistenceModel, SeasonalNaiveModel,  # noqa: E402
                              time_folds)

DATA = pd.read_csv(PROCESSED_DIR / "model_dataset.csv")
TRAIN = DATA[(DATA.t <= 18) & ~DATA.is_holdout]


def test_markov_chain_probabilities():
    m = MarkovChainModel().fit(TRAIN[FINAL_INPUT], TRAIN[TARGET])
    P = m.predict_proba(TRAIN[FINAL_INPUT].head(200))
    assert P.shape == (200, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    # a plant that was High last month is most likely High again (sticky target)
    row = TRAIN[FINAL_INPUT].iloc[[0]].assign(cf_lag1=0.9, category="Mini Hydro", season="SW Monsoon")
    assert m.predict(row)[0] == 2


def test_rule_baselines():
    X = TRAIN[FINAL_INPUT].head(50)
    assert set(PersistenceModel().fit(X).predict(X)) <= {0, 1, 2}
    no_last_year = X.assign(cf_lag12=np.nan)
    assert (SeasonalNaiveModel().fit(X).predict(no_last_year) == PersistenceModel().predict(no_last_year)).all()


def test_model_frame_handles_unknown_plants():
    f = ModelFrame(use_plant_id=True, mode="category").fit(TRAIN[FINAL_INPUT])
    Z = f.transform(TRAIN[FEATURES].head(3))          # no plant_id column at all (typed-in plant)
    assert list(Z.columns) == FINAL_INPUT
    assert Z.plant_id.isna().all()                    # unseen -> missing category, handled natively


def test_time_folds_never_look_ahead():
    for tr, va in time_folds(TRAIN, (7, 9, 11)):
        assert TRAIN.t.values[tr].max() < TRAIN.t.values[va].min()


@pytest.mark.skipif(not (FINAL_MODELS_DIR / "final_metadata.json").exists(), reason="run src.final_train first")
def test_saved_final_models_predict_holdout():
    meta = json.loads((FINAL_MODELS_DIR / "final_metadata.json").read_text())
    hold = json.loads((PROCESSED_DIR / "holdout_sample.json").read_text())
    x = pd.DataFrame([{c: hold[c] for c in FINAL_INPUT}])
    for name, fname in meta["app_models"].items():
        with open(MODELS_DIR / fname, "rb") as fh:
            p = pickle.load(fh).predict_proba(x)[0]
        assert p.shape == (3,) and abs(p.sum() - 1) < 1e-6, name
