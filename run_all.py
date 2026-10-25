"""Rebuild everything from the raw CSVs: data prep -> EDA -> training -> holdout inference -> SHAP/LIME.

    python run_all.py
"""
from src import data_prep, eda, explain_ensemble, predict, train

if __name__ == "__main__":
    print("== 1/5 data preparation ==")
    data_prep.main()
    print("\n== 2/5 exploratory figures ==")
    eda.main()
    print("\n== 3/5 training + evaluation (about 3 minutes) ==")
    train.main()
    print("\n== 4/5 single-sample inference ==")
    predict.predict_holdout()
    print("\n== 5/5 explainability: SHAP + LIME (about 4 minutes) ==")
    explain_ensemble.main()
    print("\nDone. Start the front end with:  python app/app.py")
