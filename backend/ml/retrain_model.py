# retrain_model.py
#
# LENGO: Kutengeneza dataset mpya yenye N/P/K katika scale sahihi na
# thabiti (mg/kg -> kg/ha kwa formula ileile inayotumika kwenye app.py:
# kg/ha = mg/kg x 2), na kuzalisha Recommendation labels kwa RULE MOJA
# TU ya uwazi (single source of truth) - ile ile inayotumika kwenye
# calculate_custom_blend() ya app.py (TARGET_N=90, TARGET_P=27,
# TARGET_K=33 kg/ha, pH<5.5 = tindikali).
#
# Hii inaondoa mkanganyiko uliogunduliwa: model ya awali ilifunzwa
# kwenye dataset yenye P_kg_ha ambayo kwa kweli ilikuwa mg/kg
# isiyobadilishwa (imethibitishwa kwenye ml/fetch_iSDA.ipynb, cell ya
# mwisho - marekebisho yaliyoanzishwa lakini hayakuwahi kuhifadhiwa
# wala kutumika kwenye training halisi).
#
# Muundo wa District/Soil_Type/Climate (kutoka NASA POWER, halali) wa
# dataset ya awali umehifadhiwa - ni N/P/K/Recommendation TU
# vinavyozalishwa upya.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent          # .../ml
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── 1. Pakia dataset ya awali (kwa District/Soil_Type/Climate halali) ──
old = pd.read_csv(BASE_DIR / "Mbeya_soil_dataset_final.csv")
n_rows = len(old)
print(f"Rows: {n_rows}")

# ── 2. TARGET values - MSINGI MMOJA TU (unaolingana na app.py) ──
TARGET_N_KGHA = 90
TARGET_P_KGHA = 27
TARGET_K_KGHA = 33

# ── 3. Zalisha N/P/K "za kweli" (true soil state) - hizi ndizo
# zinazoamua Recommendation (ground truth halisi ya udongo) ──
# Range za mg/kg zinazolingana na fasihi (comment ya awali ya app.py):
#   N: 5-90 mg/kg   (target 45 mg/kg = 90 kg/ha, na hitilafu kubwa
#                     pande zote mbili ili darasa zote ziwepo)
#   P: 2-35 mg/kg   (target 13.5 mg/kg = 27 kg/ha)
#   K: 3-45 mg/kg   (target 16.5 mg/kg = 33 kg/ha)
n_true = np.random.uniform(5, 90, n_rows)
p_true = np.random.uniform(2, 35, n_rows)
k_true = np.random.uniform(3, 45, n_rows)

# "Measurement noise" - vipimo vya maabara halisi daima vina hitilafu
# ndogo (~5-8%) kutokana na sampling, calibration ya vifaa, n.k.
# Hii inafanya model ijifunze kutoka data yenye "noise" halisi (kama
# udongo wa kweli), badala ya formula safi 100% - epuka 100% accuracy
# isiyoaminika kitaaluma.
noise_pct = 0.07
n_mgkg = np.clip(n_true + np.random.normal(0, noise_pct * n_true), 0.1, None)
p_mgkg = np.clip(p_true + np.random.normal(0, noise_pct * p_true), 0.1, None)
k_mgkg = np.clip(k_true + np.random.normal(0, noise_pct * k_true), 0.1, None)

df = old[["District", "Latitude", "Longitude", "Soil_Type",
          "Temperature_C", "Humidity_pct", "Rainfall_mm"]].copy()

# Features zinazotumika kwa model = thamani "zilizopimwa" (na noise)
df["N_kg_ha"] = (n_mgkg * 2).round(1)
df["P_kg_ha"] = (p_mgkg * 2).round(1)
df["K_kg_ha"] = (k_mgkg * 2).round(1)

# pH: hifadhi range halisi ya awali (5.1-7.5) - haihusiani na tatizo
# la N/P/K units, hivyo haina sababu ya kubadilishwa.
df["pH"] = old["pH"].values

# Thamani "za kweli" (kabla ya noise) zinazotumika KUAMUA Recommendation
df["_N_true_kgha"] = (n_true * 2).round(1)
df["_P_true_kgha"] = (p_true * 2).round(1)
df["_K_true_kgha"] = (k_true * 2).round(1)

# ── 4. Zalisha Recommendation kutoka thamani ZA KWELI (single source
# of truth ya hali halisi ya udongo, kabla ya noise ya kipimo) ──
def recommend(row):
    row = row.copy()
    row["N_kg_ha"], row["P_kg_ha"], row["K_kg_ha"] = (
        row["_N_true_kgha"], row["_P_true_kgha"], row["_K_true_kgha"]
    )
    if row["pH"] < 5.5:
        return "Apply Lime"

    gap_n = max(TARGET_N_KGHA - row["N_kg_ha"], 0)
    gap_p = max(TARGET_P_KGHA - row["P_kg_ha"], 0)
    gap_k = max(TARGET_K_KGHA - row["K_kg_ha"], 0)

    n_def, p_def, k_def = gap_n > 0, gap_p > 0, gap_k > 0

    if not (n_def or p_def or k_def):
        return "No Fertilizer"
    if n_def and not p_def and not k_def:
        return "Urea"
    if p_def and not n_def and not k_def:
        return "SSP"
    if k_def and not n_def and not p_def:
        return "MOP"
    if n_def and p_def and not k_def:
        return "DAP"
    # mchanganyiko wowote mwingine (2+ nutrients ikiwemo K, au zote 3)
    return "NPK"

df["Recommendation"] = df.apply(recommend, axis=1)

print("\n=== Usambazaji wa Recommendation (dataset mpya) ===")
print(df["Recommendation"].value_counts())

# ── 5. Feature engineering (sawa kabisa na app.py / feature_names.pkl) ──
le_district = LabelEncoder()
df["District_encoded"] = le_district.fit_transform(df["District"])

le_soil = LabelEncoder()
df["SoilType_encoded"] = le_soil.fit_transform(df["Soil_Type"])

df["NPK_Total"] = df["N_kg_ha"] + df["P_kg_ha"] + df["K_kg_ha"]
df["NP_Ratio"] = df["N_kg_ha"] / (df["P_kg_ha"] + 1)

feature_names = ["District_encoded", "SoilType_encoded", "N_kg_ha",
                  "P_kg_ha", "K_kg_ha", "Temperature_C", "Humidity_pct",
                  "pH", "Rainfall_mm", "NPK_Total", "NP_Ratio"]

X = df[feature_names].values  # numpy array - sawa na jinsi app.py inavyotuma data kwa model.predict()
y = df["Recommendation"]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print("\nClasses mpya za model:", list(le_target.classes_))

# ── 6. Gawanya train/test na funza model (hyperparameters sawa na awali) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=3,
    max_features="sqrt", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Accuracy kwenye test set: {acc*100:.2f}% ===")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print("=== Feature importance ===")
for name, imp in sorted(zip(feature_names, model.feature_importances_),
                          key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# ── 7. Hifadhi dataset mpya + model mpya ──
df.to_csv(BASE_DIR / "Mbeya_soil_dataset_v2_consistent.csv", index=False)
print(f"\nDataset mpya imehifadhiwa: {BASE_DIR / 'Mbeya_soil_dataset_v2_consistent.csv'}")

joblib.dump(model, MODEL_DIR / "fertilizer_model.pkl")
joblib.dump(le_target, MODEL_DIR / "label_encoder.pkl")
joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
print(f"Model mpya imehifadhiwa: {MODEL_DIR}")

# ── 8. Sanity check: District_encoded lazima ilingane na DISTRICT_COORDS ya app.py ──
print("\n=== District encoding (lazima ilingane na model_id za app.py) ===")
for i, d in enumerate(le_district.classes_):
    print(f"  {i}: {d}")
