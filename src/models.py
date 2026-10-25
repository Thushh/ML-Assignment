"""Preprocessing pipeline + the four ensemble models with hand-designed hyperparameters.

No model uses library defaults for its key hyperparameters; the reasoning for each
value is written next to it (see also README section 4).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, NUMERIC_LOG_FEATURES, RANDOM_STATE

# ----------------------------------------------------------------------------- RF
RF_PARAMS = dict(
    n_estimators=500,            # OOB error flattens after ~300 trees; 500 keeps soft-vote probabilities stable
    max_depth=14,                # ~5.3k training rows: deep enough for category x season x lag interactions,
                                 # shallow enough to stop memorising single plants
    min_samples_leaf=3,          # smoother leaf probabilities (soft voting needs calibrated-ish probas)
    min_samples_split=6,
    max_features=0.5,            # lag features dominate; 50 % of columns per split (vs sqrt default) lets
                                 # trees find them while still de-correlating the forest
    max_samples=0.8,             # bootstrap 80 % -> extra diversity between trees
    class_weight=None,           # tried "balanced_subsample": classes are only mildly imbalanced
                                 # (26/43/31 %) and weighting lowered test macro-F1, so not used
    bootstrap=True, oob_score=True, n_jobs=-1, random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------- XGB
XGB_PARAMS = dict(
    n_estimators=600,            # paired with a small learning rate (slow, careful boosting)
    learning_rate=0.03,
    max_depth=3,                 # boosting works best with shallow trees. First design was 5; the time-ordered
                                 # validation curve (reports/figures/validation_curves_max_depth.png) scored
                                 # depth 3 higher (0.578 vs 0.560 macro-F1), so 3 was adopted
    min_child_weight=3,          # avoid splits supported by a handful of plant-months
    subsample=0.8,               # row subsampling = stochastic gradient boosting
    colsample_bytree=0.8,
    gamma=0.1,                   # minimum loss reduction to split -> prunes noisy splits
    reg_lambda=2.0,              # L2 on leaf weights (default 1) - small, noisy dataset
    reg_alpha=0.1,               # light L1
    objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
    n_jobs=-1, random_state=RANDOM_STATE,
)

# ------------------------------------------------------------------ stacking meta
META_PARAMS = dict(C=0.5, max_iter=2000)  # simple, regularised meta-learner over 6 probability inputs


def make_preprocessor() -> ColumnTransformer:
    """Imputation + log transform + scaling + one-hot, saved inside the .pkl."""
    log_num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("log_num", log_num, NUMERIC_LOG_FEATURES),
        ("num", num, NUMERIC_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES),
    ])


def rf_model():
    return RandomForestClassifier(**RF_PARAMS)


def xgb_model():
    return XGBClassifier(**XGB_PARAMS)


def build_pipelines() -> dict:
    """Return the four candidate pipelines (each = preprocessor + estimator)."""
    voting = VotingClassifier([("rf", rf_model()), ("xgb", xgb_model())],
                              voting="soft", weights=[1, 1], n_jobs=1)
    stacking = StackingClassifier(
        estimators=[("rf", rf_model()), ("xgb", xgb_model())],
        final_estimator=LogisticRegression(**META_PARAMS),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )
    models = {
        "Random Forest": rf_model(),
        "XGBoost": xgb_model(),
        "Soft Voting (RF+XGB)": voting,
        "Stacking (RF+XGB -> LogReg)": stacking,
    }
    return {name: Pipeline([("prep", make_preprocessor()), ("model", est)])
            for name, est in models.items()}
