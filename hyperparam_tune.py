# """
# Hyperparameter tuning for MultiOutputRegressor(XGBRegressor)
# using GridSearchCV + KFold cross-validation.
# Saves best model in the same format as soil_calibration_model.pkl
# """

# import pandas as pd
# import numpy as np
# import joblib
# import time

# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor

# # ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
# df_in  = pd.read_csv(r"D:\Porjects\MODEL4_new\structured_input_data.csv")
# df_out = pd.read_csv(r"D:\Porjects\MODEL4_new\structured_output_data.csv")

# desired_order = [
#     'pH', 'Conductivity', 'Nitrogen', 'Phosphorus', 'Potassium',
#     'Boron', 'Copper', 'Ferrous', 'Manganese', 'Organic Carbon',
#     'Sulphur', 'Zinc', 'LAI', 'TII', 'ARI', 'RBI', 'CRI',
#     'FAI', 'MRI', 'HCI', 'C/N Ratio',
#     'Col410', 'Col435', 'Col460', 'Col485', 'Col510', 'Col535',
#     'Col560', 'Col585', 'Col610', 'Col645', 'Col680', 'Col705',
#     'Col730', 'Col760', 'Col810', 'Col860', 'Col900', 'Col940',
#     'Cap. Moist.', 'Temp', 'Moist', 'EC', 'Ph', 'Nitro', 'Posh', 'Pota'
# ]
# desired_target_order = [
#     'pH', 'Conductivity', 'Nitrogen', 'Phosphorus', 'Potassium',
#     'Boron', 'Copper', 'Ferrous', 'Manganese', 'Organic Carbon',
#     'Sulphur', 'Zinc', 'LAI', 'TII', 'ARI', 'RBI', 'CRI'
# ]

# X = df_in.drop(columns="Sample Name")[desired_order]
# y = df_out.drop(columns="Sample Name")[desired_target_order]

# print(f"X shape: {X.shape}  |  y shape: {y.shape}")

# # ─── 2. BASE ESTIMATOR ────────────────────────────────────────────────────────
# base_xgb = XGBRegressor(
#     objective='reg:squarederror',
#     random_state=42,
#     n_jobs=1,          # set to 1 here; parallelism handled by GridSearchCV
#     verbosity=0,
#     eval_metric='rmse'
# )

# multi_model = MultiOutputRegressor(estimator=base_xgb, n_jobs=-1)

# # ─── 3. PARAM GRID ────────────────────────────────────────────────────────────
# # NOTE: params for inner XGBRegressor MUST be prefixed with 'estimator__'
# param_grid = {
#     'estimator__n_estimators'    : [50, 100, 150],
#     'estimator__max_depth'       : [2, 3, 4],
#     'estimator__learning_rate'   : [0.03, 0.05, 0.1],
#     'estimator__subsample'       : [0.7, 0.8, 1.0],
#     'estimator__colsample_bytree': [0.7, 0.8, 1.0],
#     'estimator__min_child_weight': [1, 2, 3],     # XGB equivalent of min_samples_leaf
#     'estimator__reg_alpha'       : [0, 0.1],      # L1 regularization
#     'estimator__reg_lambda'      : [1, 1.5],      # L2 regularization
# }

# # ─── 4. KFOLD CV ──────────────────────────────────────────────────────────────
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # ─── 5. GRID SEARCH ───────────────────────────────────────────────────────────
# print("\nStarting GridSearchCV... (this may take several minutes)")
# start = time.time()

# grid_search = GridSearchCV(
#     estimator=multi_model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',   # works for multi-output (averaged)
#     cv=kf,
#     n_jobs=-1,
#     verbose=2,
#     refit=True,                         # auto-refit best params on full data
#     return_train_score=True
# )

# grid_search.fit(X, y)

# elapsed = time.time() - start
# print(f"\nGridSearchCV done in {elapsed/60:.1f} min")

# # ─── 6. BEST PARAMS & CV SCORE ────────────────────────────────────────────────
# print("\n=== BEST PARAMETERS ===")
# for k, v in grid_search.best_params_.items():
#     print(f"  {k}: {v}")

# best_cv_mse  = -grid_search.best_score_
# best_cv_rmse = np.sqrt(best_cv_mse)
# print(f"\nBest CV MSE  (avg over outputs): {best_cv_mse:.4f}")
# print(f"Best CV RMSE (avg over outputs): {best_cv_rmse:.4f}")

# # ─── 7. CROSS_VAL_SCORE VERIFICATION on best model ───────────────────────────
# print("\n=== CROSS_VAL_SCORE on best estimator (verification) ===")
# best_model = grid_search.best_estimator_

# cv_scores = cross_val_score(
#     best_model, X, y,
#     cv=kf,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1
# )
# cv_rmse_per_fold = np.sqrt(-cv_scores)
# print(f"Per-fold RMSE : {np.round(cv_rmse_per_fold, 4)}")
# print(f"Mean RMSE     : {cv_rmse_per_fold.mean():.4f} ± {cv_rmse_per_fold.std():.4f}")

# # R² per target on full training data (sanity check — not generalization metric)
# y_pred_full = best_model.predict(X)
# r2_per_target = r2_score(y, y_pred_full, multioutput='raw_values')
# print("\n=== R² per target (train, sanity) ===")
# for name, r2 in zip(desired_target_order, r2_per_target):
#     print(f"  {name:<18}: {r2:.4f}")

# # ─── 8. SAVE BEST MODEL ───────────────────────────────────────────────────────
# save_path = r"D:\Porjects\MODEL4_new\soil_calibration_model_tuned.pkl"
# model_dict = {
#     'model'        : best_model,
#     'feature_names': desired_order,
#     'target_names' : desired_target_order,
#     'best_params'  : grid_search.best_params_,
#     'best_cv_rmse' : best_cv_rmse,
# }
# joblib.dump(model_dict, save_path)
# print(f"\nBest model saved to: {save_path}")
# print("Done.")
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

# ───────────────── LOAD DATA ─────────────────

df_in = pd.read_csv(r"D:\Porjects\MODEL4_new\structured_input_data.csv")
df_out = pd.read_csv(r"D:\Porjects\MODEL4_new\structured_output_data.csv")

desired_order = [
    'pH', 'Conductivity', 'Nitrogen', 'Phosphorus', 'Potassium',
    'Boron', 'Copper', 'Ferrous', 'Manganese', 'Organic Carbon',
    'Sulphur', 'Zinc', 'LAI', 'TII', 'ARI', 'RBI', 'CRI',
    'FAI', 'MRI', 'HCI', 'C/N Ratio',
    'Col410', 'Col435', 'Col460', 'Col485', 'Col510', 'Col535',
    'Col560', 'Col585', 'Col610', 'Col645', 'Col680', 'Col705',
    'Col730', 'Col760', 'Col810', 'Col860', 'Col900', 'Col940',
    'Cap. Moist.', 'Temp', 'Moist', 'EC', 'Ph', 'Nitro', 'Posh', 'Pota'
]

desired_target_order = [
    'pH', 'Conductivity', 'Nitrogen', 'Phosphorus', 'Potassium',
    'Boron', 'Copper', 'Ferrous', 'Manganese', 'Organic Carbon',
    'Sulphur', 'Zinc', 'LAI', 'TII', 'ARI', 'RBI', 'CRI'
]

X = df_in.drop(columns="Sample Name")[desired_order]
y = df_out.drop(columns="Sample Name")[desired_target_order]

print("X shape:", X.shape)

# ───────────────── PARAM SPACE ─────────────────

param_dist = {
    'n_estimators': [50, 70, 100],
    'max_depth': [2, 3],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 2],
    'gamma': [0, 0.05],
    'reg_lambda': [1, 1.2]
}

random_params = list(ParameterSampler(
    param_dist,
    n_iter=100,
    random_state=42
))

best_models = {}
best_params = {}
best_rmse = {}

overall_start = time.time()

# ───────────────── TUNE EACH TARGET ─────────────────

for target in desired_target_order:

    print("\n" + "="*70)
    print("TUNING:", target)
    print("="*70)

    y_single = y[target]

    lowest_rmse = np.inf
    target_best_params = None

    start = time.time()

    for i, params in enumerate(random_params, 1):

        model = XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
            tree_method='hist',
            device='cuda',
            **params
        )

        model.fit(X, y_single)

        pred = model.predict(X)

        rmse = root_mean_squared_error(y_single, pred)

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            target_best_params = params

        if i % 10 == 0:
            print(f"{i}/100  Current best RMSE = {lowest_rmse:.5f}")

    # Train final model using best params
    final_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42,
        tree_method='hist',
        device='cuda',
        **target_best_params
    )

    final_model.fit(X, y_single)

    best_models[target] = final_model
    best_params[target] = target_best_params
    best_rmse[target] = lowest_rmse

    elapsed = time.time() - start

    print("\nBEST PARAMS:")
    print(target_best_params)
    print(f"BEST TRAIN RMSE: {lowest_rmse:.5f}")
    print(f"Time: {elapsed/60:.2f} min")

# ───────────────── SAVE ─────────────────

save_dict = {
    "models": best_models,
    "feature_names": desired_order,
    "target_names": desired_target_order,
    "best_params": best_params,
    "best_rmse": best_rmse
}

joblib.dump(
    save_dict,
    r"D:\Porjects\MODEL4_new\soil_models_full_data_tuned.pkl"
)

print("\nDONE")
print(f"Total time: {(time.time()-overall_start)/60:.2f} minutes")