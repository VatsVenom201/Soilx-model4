import joblib
import pandas as pd

# Load the existing model
model_path = r'D:\Porjects\MODEL4_new\soil_calibration_model_tuned.pkl'
try:
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    feature_names = model_dict['feature_names']
    target_names = model_dict['target_names']
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# New test data provided
test_data = {
    'pH': 8.07,
    'Conductivity': 627.0,
    'Nitrogen': 212.0872,
    'Phosphorus': 44.9963,
    'Potassium': 205.7015,
    'Boron': 0.7617,
    'Copper': 0.6601,
    'Ferrous': 4.616,
    'Manganese': 7.1826,
    'Organic Carbon': 0.441,
    'Sulphur': 43.5242,
    'Zinc': 1.3285,
    'LAI': 2.18,
    'TII': 0.5305,
    'ARI': 1.9175,
    'RBI': 4.0198,
    'CRI': 0.2151,
    'FAI': 19.5735,
    'MRI': 9.8354,
    'HCI': 0.1462,
    'C/N Ratio': 20.793,

    'Col410': 2469.275,
    'Col435': 371.145,
    'Col460': 836.96,
    'Col485': 124.64,
    'Col510': 609.03,
    'Col535': 621.785,
    'Col560': 62.805,
    'Col585': 56.82,
    'Col610': 277.715,
    'Col645': 57.0,
    'Col680': 211.21,
    'Col705': 47.23,
    'Col730': 77.57,
    'Col760': 59.72,
    'Col810': 948.39,
    'Col860': 294.465,
    'Col900': 50.64,
    'Col940': 17.14,

    'Cap. Moist.': 100.0,
    'Temp': 33.7,
    'Moist': 100.0,
    'EC': 627.0,
    'Ph': 8.07,
    'Nitro': 37.0,
    'Posh': 50.0,
    'Pota': 125.5
}

# Create dataframe using the feature names from the model
X_test = pd.DataFrame([test_data], columns=feature_names)

print("\n--- PREDICTIONS FROM LOADED MODEL ---")
try:
    preds = model.predict(X_test)[0]
    for name, val in zip(target_names, preds):
        print(f"{name:<15}: {val:.4f}")
except Exception as e:
    print(f"Error during prediction: {e}")
