"""FINAL ASSIGNMENT - rebuild everything from the raw CSVs.

    python run_final.py               full run (~25 min: tuning ~15 min + training ~10 min)
    python run_final.py --skip-tune   reuse models/final/best_params.json (~10 min)

Order: data prep -> data analysis -> tuning -> train/compare/select -> SHAP interpretation -> holdout check
(The class activity is still rebuilt with `python run_all.py`.)
"""
import sys

from src import analysis, data_prep, explain, final_train, tune

if __name__ == "__main__":
    print("== 1/5 data preparation ==")
    data_prep.main()
    print("\n== 2/5 data analysis (training period only) ==")
    analysis.main()
    if "--skip-tune" in sys.argv:
        print("\n== 3/5 tuning skipped (using models/final/best_params.json) ==")
    else:
        print("\n== 3/5 hyperparameter tuning (CatBoost, LightGBM) ==")
        tune.main()
    print("\n== 4/5 train, compare and select (10 models) ==")
    final_train.main()
    print("\n== 5/5 interpretation (SHAP, hybrid weights) ==")
    explain.main()
    print("\nDone. Report: reports/final/model_selection_report.md - website: python app/app.py")
