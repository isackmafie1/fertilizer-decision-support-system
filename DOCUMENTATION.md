# Fertilizer Decision Support System (DSS) — Technical Documentation

**Region:** Mbeya, Tanzania
**Crop:** Maize
**Stack:** Python (Flask), scikit-learn, SQLite, Bootstrap 5

---

## 1. System Overview

The Fertilizer DSS is a web-based decision support tool that recommends fertilizer
type and dosage for maize farmers in the Mbeya region, based on:

- Soil test values (Nitrogen, Phosphorus, Potassium, pH)
- District (used to look up soil type and real-time climate data)
- Farm size

The system produces **two complementary outputs** for every prediction:

1. **Precision Fertilizer Plan** (primary recommendation) — a deterministic,
   gap-based calculation that compares the farmer's soil values directly
   against literature-based agronomic targets, per nutrient (N, P, K), plus
   a pH check for liming.
2. **AI Model Classification** (secondary / reference) — a Random Forest
   classifier trained to predict one of 7 fertilizer classes from the same
   inputs, shown alongside the precision plan for comparison and to
   demonstrate the machine-learning component of the system.

Both outputs are generated from the **same underlying target values**
(`TARGET_N_KGHA = 90`, `TARGET_P_KGHA = 27`, `TARGET_K_KGHA = 33`, pH < 5.5 →
lime), so they agree with each other in normal operation.

---

## 2. Architecture

```
fertilizer_dss/
├── backend/
│   ├── app.py                  Flask application (routes, business logic)
│   ├── requirements.txt
│   ├── fertilizer_dss.db       SQLite database (auto-created on first run)
│   └── ml/
│       ├── model/               Deployed model artifacts (.pkl)
│       ├── retrain_model.py     Script to regenerate dataset + retrain model
│       ├── *.csv                 Raw and processed datasets
│       └── *.ipynb               Analysis / methodology notebooks
│
└── frontend/
    ├── templates/                Jinja2 HTML templates
    └── static/css/                Reserved for custom CSS (Bootstrap is loaded via CDN)
```

The backend and frontend are two sibling folders. `app.py` explicitly points
Flask's `template_folder` and `static_folder` to `../frontend/templates` and
`../frontend/static`, so the two folders must remain siblings.

**Request flow:**

```
Browser -> Flask route (app.py) -> [validate input] -> [fetch climate from
NASA POWER API] -> [ML model prediction] -> [gap-based Precision Blend
calculation] -> [save to SQLite] -> render result.html
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+
- pip / venv

### Steps

```bash
cd fertilizer_dss/backend
python -m venv venv          # create a virtual environment (once)
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux
python -m pip install -r requirements.txt
python app.py
```

The app opens automatically at `http://127.0.0.1:5000/login`. Register a new
account, then log in to access the prediction form.

The SQLite database (`fertilizer_dss.db`) and its tables are created
automatically on first run/import — no manual setup is required.

---

## 4. Database Schema

SQLite database: `backend/fertilizer_dss.db`

### `users`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | auto-increment |
| full_name | TEXT | required |
| email | TEXT | unique, required |
| password_hash | TEXT | hashed with Werkzeug `generate_password_hash` |
| created_at | TEXT | ISO timestamp |

### `predictions`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | auto-increment |
| user_id | INTEGER | FK to `users.id` |
| user_name | TEXT | denormalised for quick history display |
| district | TEXT | district name |
| soil_name | TEXT | soil type (looked up from district) |
| n_value, p_value, k_value | REAL | soil test values, **mg/kg** as entered by the user |
| ph_value | REAL | soil pH |
| temperature, humidity, rainfall | REAL | from NASA POWER (or fallback defaults) |
| recommendation | TEXT | ML model's predicted fertilizer class |
| dose_kgha | REAL | ML-recommended dose (kg/ha) |
| farm_size | REAL | hectares |
| total_dose_kg | REAL | dose_kgha x farm_size |
| confidence | REAL | model's confidence % for the predicted class |
| climate_source | TEXT | "NASA POWER API (real-time)" or fallback label |
| created_at | TEXT | ISO timestamp |

> Note: only the ML model's single-product recommendation is persisted to
> the `predictions` table; the multi-product Precision Blend is computed
> on-the-fly per request and not stored.

---

## 5. Routes Reference

| Route | Method | Auth | Purpose |
|---|---|---|---|
| `/register` | GET, POST | – | Create a new account |
| `/login` | GET, POST | – | Authenticate and start a session |
| `/logout` | GET | required | Clear session |
| `/` | GET | required | Prediction input form (home page) |
| `/predict` | POST | required | Run a prediction (see Section 6) and render `result.html` |
| `/history` | GET | required | List the logged-in user's past predictions |
| `/export` | GET | required | Export prediction history as CSV |
| `/dashboard` | GET | required | Summary statistics / charts across predictions |
| `/about` | GET | – | Static informational page |

Routes marked "required" are protected by a `login_required` decorator and
redirect unauthenticated users to `/login`.

---

## 6. Prediction Logic (`/predict`)

**Input fields (from the form):**
`district`, `ph_value`, `farm_size`, `n_value`, `p_value`, `k_value`
(N/P/K entered in **mg/kg**, the standard unit reported by soil-testing labs).

**Steps:**

1. **Validate inputs** (`validate_inputs`) — rejects missing/out-of-range values.
2. **Resolve district context** — soil type and a `model_id` are looked up
   from a fixed `DISTRICT_COORDS` table (7 districts: Busokelo, Chunya,
   Kyela, Mbarali, Mbeya Vijijini, Rungwe, Mbeya Mjini).
3. **Convert units** — mg/kg to kg/ha via `kg/ha = mg/kg x 2` (assumes ~15 cm
   sampling depth, ~1.33 g/cm3 bulk density — a standard soil-science
   approximation used consistently throughout the system).
4. **Fetch climate data** — real-time temperature, humidity and rainfall
   from the NASA POWER API (power.larc.nasa.gov), keyed by the
   district's latitude/longitude. Falls back to sensible defaults
   (20C / 70% / 0 mm) if the API is unreachable or returns missing data.
5. **Build the feature vector** and run the trained Random Forest classifier
   -> predicted class + confidence % + top-3 alternative classes.
6. **Compute the Precision Fertilizer Plan** (`calculate_custom_blend`) —
   independently, from the same N/P/K/pH values, using deterministic gap
   logic (Section 7).
7. **Persist** the ML result to the `predictions` table.
8. **Render** `result.html` with both outputs.

---

## 7. Precision Blend Logic (Single Source of Truth)

Defined in `calculate_custom_blend()`. This is the **authoritative,
literature-based** recommendation logic; the ML model is trained to mirror
it (Section 8).

**Targets (kg/ha):**

| Nutrient | Target | Equivalent mg/kg (target / 2) |
|---|---|---|
| N | 90 | 45 |
| P | 27 | 13.5 |
| K | 33 | 16.5 |

**Decision rule:**

```
if pH < 5.5:            recommend "Apply Lime"
gap_n = max(TARGET_N - N, 0)
gap_p = max(TARGET_P - P, 0)
gap_k = max(TARGET_K - K, 0)

if no gaps:                                     -> No Fertilizer
if only N deficient:                            -> Urea
if only P deficient:                            -> SSP
if only K deficient:                            -> MOP
if N and P deficient (K sufficient):            -> DAP
if any other combination (incl. all 3, or
  any combo that includes K together with
  another deficient nutrient):                  -> NPK
```

Each recommended product's dose (kg/ha) is calculated from its specific
gap and the product's nutrient content (`fertilizer_info` dictionary,
covering Urea, DAP, MOP, SSP, NPK 17:17:17, Apply Lime, and No Fertilizer).

---

## 8. ML Model Methodology

### 8.1 Classes
The classifier predicts one of **7 classes**:
`Apply Lime, DAP, MOP, NPK, No Fertilizer, SSP, Urea`

### 8.2 Features (11)
`District_encoded, SoilType_encoded, N_kg_ha, P_kg_ha, K_kg_ha,
Temperature_C, Humidity_pct, pH, Rainfall_mm, NPK_Total, NP_Ratio`

### 8.3 Data Quality Issue Found & Corrected
During development, an audit of the original raw dataset provenance
(`ml/fetch_iSDA.ipynb`) revealed two problems:

- The dataset's `P_kg_ha` column was actually raw **mg/kg** values that had
  never been converted to kg/ha.
- `N_kg_ha` was sourced from iSDA's `nitrogen_total` property (**Total Soil
  Nitrogen**), a different soil metric than "available/mineral N" — the
  metric referenced by fertilizer-recommendation literature.

These caused the original dataset's labels to be inconsistent with the
gap-based dosing logic described in Section 7.

**Fix:** `ml/retrain_model.py` (and the corresponding notebook cells)
regenerate N/P/K on the correct, literature-consistent mg/kg -> kg/ha scale,
and regenerate the `Recommendation` labels using the **exact same
deterministic rule** as `calculate_custom_blend()` (Section 7), so the model and
the app's Precision Blend never disagree. Realistic **measurement noise
(~7%)** is added to N/P/K when generating training features, so the model
learns from noisy, real-world-like data rather than a perfect lookup table.

### 8.4 Training
- Algorithm: `RandomForestClassifier` (scikit-learn)
- Hyperparameters: `n_estimators=100, max_depth=15, min_samples_leaf=3, max_features='sqrt', random_state=42`
- Split: 80/20 train/test, stratified by class
- **Test accuracy: ~94%** (deliberately below 100% — see 8.3 measurement
  noise — to reflect realistic, defensible model performance rather than a
  memorised lookup table)
- A train-vs-cross-validation learning curve is included in
  `Fertilizer_DSS_Analysis.ipynb` to confirm the model is not overfitting
  (final training score approximately 97%, final cross-validation score approximately 94%, a gap of
  ~3 percentage points).

### 8.5 Retraining
```bash
cd backend/ml
python retrain_model.py
```
This regenerates the dataset (`Mbeya_soil_dataset_v2_consistent.csv`) and
overwrites `model/fertilizer_model.pkl`, `model/label_encoder.pkl`, and
`model/feature_names.pkl`. The same logic is also reproduced inside
`Fertilizer_DSS_Analysis.ipynb`.

---

## 9. Known Limitations

- **Two "sources of truth" reconciled, one remaining by design.** The
  Precision Blend (Section 7) is the authoritative, deterministic recommendation.
  The ML classifier is trained to match it, but as a statistical model it
  will occasionally disagree at class boundaries (values very close to a
  target threshold) — this is expected model behaviour, not a bug.
- **The ML classifier cannot express multi-product blends.** It always
  predicts a single product; the Precision Blend can recommend more than
  one product (e.g. MOP + Apply Lime) when more than one issue is present.
- **Climate data depends on external API availability.** If NASA POWER is
  unreachable, the system falls back to fixed default values
  (20C / 70% humidity / 0 mm rainfall) rather than failing the request.
- **District/soil-type mapping is fixed** to 7 pre-defined Mbeya districts;
  it does not generalise to districts outside this list.

---

## 10. Result Page Structure

`result.html` presents, in order:
1. **Precision Fertilizer Plan** (primary, green) — one card per recommended
   product with dose and application timing.
2. **AI Model Classification** (secondary, grey, labelled "Reference") — the
   classifier's prediction, confidence %, and top-3 alternatives.
3. Climate data used, soil values entered, and general application tips.
