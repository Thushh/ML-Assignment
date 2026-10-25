"""FINAL ASSIGNMENT - model definitions.

Three groups of models are compared (README, Part 2):

1. Existing approaches   - what a planner or a statistician would use today:
                           persistence, seasonal naive, Markov chain, multinomial logistic regression
2. Activity models       - Random Forest, XGBoost, Soft Voting (src/models.py, reused unchanged)
3. New models            - CatBoost, LightGBM, and a hybrid stack (Markov chain + LightGBM + RF)

Every model is an sklearn-compatible object with fit / predict / predict_proba that takes the
SAME input DataFrame (FEATURES + plant_id), so they can be cross-validated, compared, pickled
and served by the web app in exactly the same way.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import CATEGORICAL_FEATURES, FEATURES, PLANT_ID, RANDOM_STATE
from .features import cf_to_class
from .models import build_pipelines, make_preprocessor, rf_model

CLASSES = np.array([0, 1, 2])


def previous_class(X: pd.DataFrame) -> np.ndarray:
    """Last month's class (falls back to the 3-month mean, then a typical CF of 0.2)."""
    return X["cf_lag1"].fillna(X["cf_roll3"]).fillna(0.2).apply(cf_to_class).values


# =============================================================================
# 1. EXISTING APPROACHES
# =============================================================================
class _RuleModel(BaseEstimator, ClassifierMixin):
    """Base class for rules that need no learning (they still expose the sklearn API)."""
    deterministic = True

    def fit(self, X, y=None):
        self.classes_ = CLASSES
        return self

    def _rule(self, X):
        raise NotImplementedError

    def predict(self, X):
        return self._rule(X)

    def predict_proba(self, X):
        return np.eye(3)[self._rule(X)]        # 100 % on the predicted class


class PersistenceModel(_RuleModel):
    """'Next month = this month': the standard naive benchmark in energy forecasting."""

    def _rule(self, X):
        return previous_class(X)


class SeasonalNaiveModel(_RuleModel):
    """'Next month = the same month last year' (falls back to persistence when no last year)."""

    def _rule(self, X):
        lag12 = X["cf_lag12"]
        seasonal = lag12.fillna(0).apply(cf_to_class).values
        return np.where(lag12.isna().values, previous_class(X), seasonal)


class MarkovChainModel(BaseEstimator, ClassifierMixin):
    """First-order Markov chain over the three output states.

    P(class this month | class last month, technology, season), estimated by counting
    transitions in the training data. ARIMA-type models suit continuous series; for a
    DISCRETE state sequence the classical statistical time-series model is a Markov chain.

    Sparse cells back off to coarser conditions:
        (last class, technology, season) -> (last class, technology) -> (last class)
    and Laplace smoothing (`alpha`) avoids zero probabilities.
    """
    deterministic = False

    def __init__(self, alpha=1.0, min_count=10):
        self.alpha = alpha          # Laplace smoothing pseudo-count
        self.min_count = min_count  # a cell needs >= this many transitions to be trusted

    def fit(self, X, y):
        d = pd.DataFrame({"prev": previous_class(X), "category": X["category"].values,
                          "season": X["season"].values, "y": np.asarray(y)})
        self.levels_ = []
        for keys in (["prev", "category", "season"], ["prev", "category"], ["prev"]):
            counts = d.groupby(keys + ["y"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
            counts = counts[counts.sum(axis=1) >= self.min_count]
            probs = (counts + self.alpha).div((counts + self.alpha).sum(axis=1), axis=0)
            self.levels_.append((keys, {k if isinstance(k, tuple) else (k,): v.values
                                        for k, v in probs.iterrows()}))
        prior = np.bincount(np.asarray(y), minlength=3) + self.alpha
        self.prior_ = prior / prior.sum()
        self.classes_ = CLASSES
        return self

    def predict_proba(self, X):
        d = pd.DataFrame({"prev": previous_class(X), "category": X["category"].values,
                          "season": X["season"].values})
        out = np.tile(self.prior_, (len(d), 1))
        done = np.zeros(len(d), dtype=bool)
        for keys, table in self.levels_:
            for i, row in enumerate(d[keys].itertuples(index=False, name=None)):
                if not done[i] and row in table:
                    out[i], done[i] = table[row], True
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# Multinomial logistic regression: the classical statistical classifier (linear decision boundaries)
LOGREG_PARAMS = dict(
    C=0.5,              # moderate L2 regularisation: lag features are strongly correlated (see analysis)
    max_iter=3000,
)


def logistic_pipeline():
    return Pipeline([("prep", make_preprocessor()), ("model", LogisticRegression(**LOGREG_PARAMS))])


# =============================================================================
# 3. NEW MODELS
# =============================================================================
class ModelFrame(BaseEstimator, TransformerMixin):
    """Select the model's columns and give categorical columns the dtype the library expects.

    * mode="category": pandas 'category' dtype with categories fixed at fit time (LightGBM)
    * mode="str":      plain strings (CatBoost)
    Missing numeric values are left as NaN: both libraries handle missing values natively
    (they learn which branch missing values should follow), so no imputation is needed.
    """

    def __init__(self, use_plant_id=False, mode="category"):
        self.use_plant_id = use_plant_id
        self.mode = mode

    def _cols(self):
        return FEATURES + ([PLANT_ID] if self.use_plant_id else [])

    def _cats(self):
        return CATEGORICAL_FEATURES + ([PLANT_ID] if self.use_plant_id else [])

    def fit(self, X, y=None):
        X = self._ensure_plant(X)
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in self._cats()}
        return self

    def _ensure_plant(self, X):
        if self.use_plant_id and PLANT_ID not in X.columns:
            X = X.assign(**{PLANT_ID: "unknown"})
        return X

    def transform(self, X):
        Z = self._ensure_plant(X)[self._cols()].copy()
        for c in Z.columns.difference(self._cats()):      # None / text -> float NaN (LightGBM needs numeric dtypes)
            Z[c] = pd.to_numeric(Z[c], errors="coerce").astype(float)
        for c in self._cats():
            if self.mode == "category":   # unseen values -> NaN -> treated as missing by LightGBM
                Z[c] = pd.Categorical(Z[c].astype(str), categories=self.categories_[c])
            else:
                Z[c] = Z[c].astype(str)
        return Z


# ---------------------------------------------------------------------------- CatBoost
# Designed starting values; src/tune.py refines them with time-series CV and the result
# (models/final/best_params.json) overrides these defaults.
CATBOOST_PARAMS = dict(
    iterations=800,           # paired with a small learning rate
    learning_rate=0.03,
    depth=6,                  # CatBoost grows symmetric (oblivious) trees: depth 6 = 64 leaves,
                              # enough for technology x season x lag interactions
    l2_leaf_reg=5.0,          # stronger than default (3): ~5k noisy rows
    loss_function="MultiClass",
    random_seed=RANDOM_STATE,
    verbose=0,
    thread_count=-1,
    allow_writing_files=False,
    use_plant_id=True,        # 331-level categorical handled by ordered target statistics
)

# ---------------------------------------------------------------------------- LightGBM
LIGHTGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=15,            # LightGBM grows leaf-wise; capping leaves (and depth) prevents the
    max_depth=5,              # deep, lopsided trees that overfit small data
    min_child_samples=30,     # a leaf must represent >= 30 plant-months
    subsample=0.8, subsample_freq=1,   # bagging every iteration
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
    use_plant_id=False,       # native categorical splits on 331 levels tend to overfit (tested in tune.py)
)


class CatBoostModel(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper: CatBoost changes its `cat_features` argument internally, which
    breaks sklearn's clone() (needed by cross-validation and stacking). Creating the
    CatBoostClassifier inside fit() keeps the wrapper's parameters untouched."""

    def __init__(self, params=None, cat_features=()):
        self.params = params
        self.cat_features = cat_features

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(cat_features=list(self.cat_features), **(self.params or {}))
        self.model_.fit(X, np.asarray(y))
        self.classes_ = CLASSES
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def catboost_pipeline(params=None):
    p = dict(CATBOOST_PARAMS, **(params or {}))
    use_plant = p.pop("use_plant_id")
    cats = tuple(CATEGORICAL_FEATURES + ([PLANT_ID] if use_plant else []))
    return Pipeline([("frame", ModelFrame(use_plant_id=use_plant, mode="str")),
                     ("model", CatBoostModel(params=p, cat_features=cats))])


def lightgbm_pipeline(params=None):
    p = dict(LIGHTGBM_PARAMS, **(params or {}))
    use_plant = p.pop("use_plant_id")
    return Pipeline([("frame", ModelFrame(use_plant_id=use_plant, mode="category")),
                     ("model", LGBMClassifier(**p))])


def hybrid_stack(lgbm_params=None):
    """Hybrid statistical + ML stack.

    Base models : Markov chain (statistical time-series model of the state sequence)
                  LightGBM (gradient boosting) and Random Forest (bagging)
    Meta-learner: multinomial logistic regression trained on the base models' OUT-OF-FOLD
                  probabilities (5-fold), so it learns how much to trust each model without
                  the target leakage we observed when Markov probabilities were fed in directly.
    """
    rf_pipe = Pipeline([("prep", make_preprocessor()), ("model", rf_model())])
    return StackingClassifier(
        estimators=[("markov", MarkovChainModel()),
                    ("lgbm", lightgbm_pipeline(lgbm_params)),
                    ("rf", rf_pipe)],
        final_estimator=LogisticRegression(C=0.5, max_iter=2000),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )


# =============================================================================
# THE FULL LINE-UP
# =============================================================================
GROUPS = {
    "Persistence (same as last month)": "Existing approach",
    "Seasonal naive (same month last year)": "Existing approach",
    "Markov chain": "Existing approach",
    "Logistic regression": "Existing approach",
    "Random Forest": "Activity model",
    "XGBoost": "Activity model",
    "Soft Voting (RF+XGB)": "Activity model",
    "CatBoost": "New model",
    "LightGBM": "New model",
    "Hybrid stack (Markov+LightGBM+RF)": "New model",
}


def build_final_models(best_params=None) -> dict:
    """All 10 candidates, untrained. `best_params` comes from src/tune.py (optional)."""
    bp = best_params or {}
    activity = build_pipelines()
    return {
        "Persistence (same as last month)": PersistenceModel(),
        "Seasonal naive (same month last year)": SeasonalNaiveModel(),
        "Markov chain": MarkovChainModel(),
        "Logistic regression": logistic_pipeline(),
        "Random Forest": activity["Random Forest"],
        "XGBoost": activity["XGBoost"],
        "Soft Voting (RF+XGB)": activity["Soft Voting (RF+XGB)"],
        "CatBoost": catboost_pipeline(bp.get("CatBoost")),
        "LightGBM": lightgbm_pipeline(bp.get("LightGBM")),
        "Hybrid stack (Markov+LightGBM+RF)": hybrid_stack(bp.get("LightGBM")),
    }


def is_deterministic(model) -> bool:
    """Rules output 0/1 'probabilities': log-loss and ROC-AUC are not meaningful for them."""
    return getattr(model, "deterministic", False)


# =============================================================================
# EVALUATION HELPER
# =============================================================================
def time_folds(df, starts):
    """Expanding-window time-series CV: for each start s, train on months < s, validate on s, s+1.

    Returns a list of (train_positions, validation_positions) for `df` (positional indices).
    """
    t = df["t"].values
    return [(np.where(t < s)[0], np.where((t >= s) & (t < s + 2))[0]) for s in starts]
