import pandas as pd
import numpy as np
import joblib
import time
from itertools import product
from sklearn.metrics import root_mean_squared_error
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

# ───────────────── PARAM GRID ─────────────────

param_grid = {
    'n_estimators': [50, 70, 100, 150, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.05, 0.1],
    'reg_alpha': [0, 0.01, 0.05, 0.1],
    'reg_lambda': [0.8, 1.0, 1.2, 1.5]
}

from sklearn.model_selection import ParameterSampler

all_combinations = list(
    ParameterSampler(
        param_grid,
        n_iter=3000,
        random_state=42
    )
)

print(f"Total combinations = {len(all_combinations):,}")

best_models = {}
best_params = {}
best_rmse = {}

overall_start = time.time()

# ───────────────── LOOP OVER TARGETS ─────────────────

for target in desired_target_order:

    print("\n" + "="*80)
    print("TUNING:", target)
    print("="*80)

    y_single = y[target]

    lowest_rmse = np.inf
    target_best_params = None

    target_start = time.time()

    for i, params in enumerate(all_combinations, 1):

        model = XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
            **params
        )

        model.fit(X, y_single)

        pred = model.predict(X)

        rmse = root_mean_squared_error(y_single, pred)

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            target_best_params = params

            print(f"\nNEW BEST FOR {target}")
            print(f"RMSE = {lowest_rmse:.6f}")
            print(target_best_params)

        if i % 1000 == 0:
            print(
                f"{target}: {i:,}/{len(all_combinations):,} "
                f"current best RMSE={lowest_rmse:.6f}"
            )

    # Final clean model
    final_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42,
        **target_best_params
    )

    final_model.fit(X, y_single)

    best_models[target] = final_model
    best_params[target] = target_best_params
    best_rmse[target] = lowest_rmse

    elapsed = (time.time() - target_start)/60
    print(f"\nFinished {target}")
    print(f"Best RMSE = {lowest_rmse:.6f}")
    print(f"Time = {elapsed:.2f} min")

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
    r"D:\Porjects\MODEL4_new\soil_models_full_grid_full_data.pkl"
)

print("\nDONE")
print("Total time:", (time.time()-overall_start)/60, "minutes")