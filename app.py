# Import libraries
from flask import Flask, render_template, request, Response, redirect, url_for, session, flash
import joblib
import numpy as np
import csv
import io
import requests
import webbrowser
import threading
import sqlite3
import os
import secrets
from pathlib import Path
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml" / "model"
DB_FILE = BASE_DIR / "fertilizer_dss.db"

# Create Flask app
app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent.parent / "frontend" / "templates"),
    static_folder=str(Path(__file__).resolve().parent.parent / "frontend" / "static"),
)
# Set DSS_SECRET_KEY in your environment for a stable key. If it is not set,
# a random key is generated at startup (users must log in again after a restart).
app.secret_key = os.environ.get("DSS_SECRET_KEY") or secrets.token_hex(32)

# Load our trained model and encoders
model = joblib.load(MODEL_DIR / "fertilizer_model.pkl")
le_target = joblib.load(MODEL_DIR / "label_encoder.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_name TEXT,
            district TEXT,
            soil_name TEXT,
            n_value REAL,
            p_value REAL,
            k_value REAL,
            ph_value REAL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            recommendation TEXT,
            dose_kgha REAL,
            farm_size REAL,
            total_dose_kg REAL,
            confidence REAL,
            climate_source TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def migrate_db():
    """
    Inaongeza columns mpya kwenye database zilizopo
    bila kufuta data yoyote iliyopo.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(predictions)")
    existing_columns = [row[1] for row in cur.fetchall()]

    new_columns = [
        ("user_id",        "INTEGER"),
        ("user_name",      "TEXT"),
        ("soil_name",      "TEXT"),
        ("farm_size",      "REAL"),
        ("total_dose_kg",  "REAL"),
        ("confidence",     "REAL"),
        ("climate_source", "TEXT"),
    ]

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            cur.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def save_prediction(data):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
        (user_id, user_name, district, soil_name, n_value, p_value, k_value,
         ph_value, temperature, humidity, rainfall, recommendation, dose_kgha,
         farm_size, total_dose_kg, confidence, climate_source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["user_id"], data["user_name"], data["district"], data["soil_name"],
        data["n_value"], data["p_value"], data["k_value"], data["ph_value"],
        data["temperature"], data["humidity"], data["rainfall"],
        data["recommendation"], data["dose_kgha"], data["farm_size"],
        data["total_dose_kg"], data["confidence"], data["climate_source"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_all_predictions(user_id=None):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if user_id:
        cur.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
            (user_id,)
        )
    else:
        cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recommendation_counts(user_id=None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    if user_id:
        cur.execute("""
            SELECT recommendation, COUNT(*) as count
            FROM predictions WHERE user_id = ?
            GROUP BY recommendation
        """, (user_id,))
    else:
        cur.execute("""
            SELECT recommendation, COUNT(*) as count
            FROM predictions GROUP BY recommendation
        """)
    rows = cur.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_district_counts(user_id=None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    if user_id:
        cur.execute("""
            SELECT district, COUNT(*) as count
            FROM predictions WHERE user_id = ?
            GROUP BY district ORDER BY count DESC
        """, (user_id,))
    else:
        cur.execute("""
            SELECT district, COUNT(*) as count
            FROM predictions
            GROUP BY district ORDER BY count DESC
        """)
    rows = cur.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_total_count(user_id=None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    if user_id:
        cur.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id = ?",
            (user_id,)
        )
    else:
        cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]
    conn.close()
    return total


def create_user(full_name, email, password):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    password_hash = generate_password_hash(password)
    try:
        cur.execute("""
            INSERT INTO users (full_name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            full_name, email.lower().strip(), password_hash,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success


def verify_user(email, password):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE email = ?",
        (email.lower().strip(),)
    )
    user = cur.fetchone()
    conn.close()
    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# Fertilizer details dictionary
fertilizer_info = {
    "Urea": {
        "npk": "46-0-0",
        "n_pct": 0.46, "p_pct": 0.0, "k_pct": 0.0,
        "when": "Apply 2-3 weeks after planting (top dressing)",
        "note": "High nitrogen content. Best for vegetative growth stage."
    },
    "DAP": {
        "npk": "18-46-0",
        "n_pct": 0.18, "p_pct": 0.46, "k_pct": 0.0,
        "when": "Apply at planting time (basal application)",
        "note": "High phosphorus. Promotes root development and early growth."
    },
    "CAN": {
        "npk": "26-0-0",
        "n_pct": 0.26, "p_pct": 0.0, "k_pct": 0.0,
        "when": "Apply 3-4 weeks after planting (top dressing)",
        "note": "Calcium Ammonium Nitrate. Suitable for acidic soils."
    },
    "NPK 17:17:17": {
        "npk": "17-17-17",
        "n_pct": 0.17, "p_pct": 0.17, "k_pct": 0.17,
        "when": "Apply at planting time (basal application)",
        "note": "Balanced fertilizer. Good for soils with deficiency in all nutrients."
    },
    "NPK": {
        "npk": "17-17-17",
        "n_pct": 0.17, "p_pct": 0.17, "k_pct": 0.17,
        "when": "Apply at planting time (basal application)",
        "note": "Balanced fertilizer. Good for soils with deficiency in all nutrients."
    },
    "MOP": {
        "npk": "0-0-60",
        "n_pct": 0.0, "p_pct": 0.0, "k_pct": 0.60,
        "when": "Apply at planting or early growth stage",
        "note": "Muriate of Potash. Best for soils deficient in potassium."
    },
    "SSP": {
        "npk": "0-20-0",
        "n_pct": 0.0, "p_pct": 0.20, "k_pct": 0.0,
        "when": "Apply at planting time (basal application)",
        "note": "Single Super Phosphate. Also supplies calcium and sulphur."
    },
    "Apply Lime": {
        "npk": "-",
        "n_pct": 0.0, "p_pct": 0.0, "k_pct": 0.0,
        "when": "Apply 2-4 weeks before planting",
        "note": "Corrects soil acidity (pH below 5.5). Mix thoroughly into topsoil."
    },
    "No Fertilizer": {
        "npk": "-",
        "n_pct": 0.0, "p_pct": 0.0, "k_pct": 0.0,
        "when": "No application needed",
        "note": "Soil nutrients are adequate for maize production. Monitor soil regularly."
    },
}

default_info = {
    "npk": "—",
    "n_pct": 0.0, "p_pct": 0.0, "k_pct": 0.0,
    "when": "Consult your extension officer",
    "note": "Follow recommended application guidelines for best results."
}

# TARGET values (kg/ha) zilizooanishwa na fasihi ya kitaalamu
# (Maize Production Manual Tanzania / literature review):
#   N:  30-60 mg/kg  -> 60-120 kg/ha  -> midpoint ~90 kg/ha
#   P:  7-20  mg/kg  -> 14-40 kg/ha   -> midpoint ~27 kg/ha
#   K:  8-25  mg/kg  -> 16-50 kg/ha   -> midpoint ~33 kg/ha
TARGET_N_KGHA = 90
TARGET_P_KGHA = 27
TARGET_K_KGHA = 33

BASE_DOSES = {
    "Urea": 75,
    "DAP": 125,
    "CAN": 110,
    "NPK 17:17:17": 150,
    "NPK": 150,
    "MOP": 60,
    "SSP": 150,
}

# Mipaka ya kiuhalisia/kiusalama ya dose ya bidhaa (kg/ha) - agronomic safety caps
MIN_PRODUCT_DOSE_KGHA = 25
MAX_PRODUCT_DOSE_KGHA = 300

VALIDATION_RULES = {
    "ph_value":  {"min": 0.0,  "max": 14.0,    "label": "Soil pH"},
    "n_value":   {"min": 0.01, "max": 1000.0,  "label": "Nitrogen (N)"},
    "p_value":   {"min": 0.01, "max": 1000.0,  "label": "Phosphorus (P)"},
    "k_value":   {"min": 0.01, "max": 1000.0,  "label": "Potassium (K)"},
    "farm_size": {"min": 0.1,  "max": 10000.0, "label": "Farm Size"},
}

def validate_inputs(form_data):
    errors = []
    for field, rules in VALIDATION_RULES.items():
        raw_value = form_data.get(field, "")
        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            errors.append(f"{rules['label']} must be a valid number.")
            continue
        if value < rules["min"] or value > rules["max"]:
            errors.append(
                f"{rules['label']} must be between {rules['min']} and "
                f"{rules['max']}. You entered {value}."
            )
    return errors


def calculate_dose(fertilizer_name, n_kgha, p_kgha, k_kgha):
    """
    Gap-based precision dose calculation.

    Badala ya kutumia fixed ratio iliyofungwa (clamp 0.7-2.0) ambayo
    haizingatii "gap" halisi kati ya kiwango cha udongo na target,
    formula hii inahesabu moja kwa moja ni kg ngapi za nutrient
    zinahitajika kufika target (kg/ha), kisha inabadilisha hiyo
    kuwa kiasi cha bidhaa ya mbolea (product dose) kwa kutumia asilimia
    ya nutrient iliyomo kwenye bidhaa husika (n_pct/p_pct/k_pct).

    dose_kgha (bidhaa) = gap_kgha (nutrient inayohitajika) / pct (asilimia ya nutrient kwenye bidhaa)

    Matokeo yanafungwa (clamped) tu kwa mipaka ya kiuhalisia/kiusalama
    ya application shambani (MIN/MAX_PRODUCT_DOSE_KGHA), siyo kwa ratio
    ya kubahatisha - hivyo dose inaendana moja kwa moja na jinsi udongo
    ulivyo mbaya (bigger gap -> bigger dose, proportionally).
    """
    info = fertilizer_info.get(fertilizer_name, default_info)
    if fertilizer_name in ["Apply Lime", "No Fertilizer"]:
        return None

    base_dose = BASE_DOSES.get(fertilizer_name, 100)

    # Chagua nutrient kuu inayolengwa na fertilizer hii, pamoja na target
    # yake (kg/ha), kiwango kilichopo udongoni (kg/ha), na asilimia yake
    # kwenye bidhaa (pct).
    if info["n_pct"] >= info["p_pct"] and info["n_pct"] >= info["k_pct"] and info["n_pct"] > 0:
        target, actual, pct = TARGET_N_KGHA, n_kgha, info["n_pct"]
    elif info["p_pct"] >= info["k_pct"] and info["p_pct"] > 0:
        target, actual, pct = TARGET_P_KGHA, p_kgha, info["p_pct"]
    elif info["k_pct"] > 0:
        target, actual, pct = TARGET_K_KGHA, k_kgha, info["k_pct"]
    else:
        # Fertilizer isiyo na nutrient % iliyoainishwa (mfano fallback)
        return round(base_dose, 1)

    # Gap halisi (kg/ha) ya nutrient inayohitajika kufika target.
    # Haiwezi kuwa hasi (udongo tayari uko sawa au juu ya target).
    gap_kgha = max(target - actual, 0)

    if gap_kgha == 0:
        # Udongo tayari uko sawa/juu ya target - dose ndogo ya matengenezo tu
        dose_kgha = MIN_PRODUCT_DOSE_KGHA
    else:
        dose_kgha = gap_kgha / pct

    # Mipaka ya kiusalama/kiuhalisia ya application shambani
    dose_kgha = max(MIN_PRODUCT_DOSE_KGHA, min(dose_kgha, MAX_PRODUCT_DOSE_KGHA))

    return round(dose_kgha, 1)


def calculate_custom_blend(ph_value, n_kgha, p_kgha, k_kgha):
    """
    Precision Custom Blend (Split-Application) Recommendation.

    Badala ya kutoa bidhaa MOJA ya fixed-ratio (mfano NPK 17:17:17)
    ambayo huenda isilingane kikamilifu na uwiano halisi wa upungufu
    wa kila nutrient, function hii inahesabu gap ya kila nutrient
    (N, P, K) KIVYAKE, kisha inapendekeza mchanganyiko wa bidhaa
    "straight" (Urea kwa N, SSP kwa P, MOP kwa K) - kila moja kwa
    kiasi kinacholingana hasa na upungufu wake.

    Hii ndiyo njia inayotumika na wataalamu wa udongo (soil agronomists)
    - "split application" / "custom blend" - kinyume na "blanket
    recommendation" ya bidhaa moja tu.

    Inarudisha list ya dictionaries, kila moja ikiwa na:
        product, nutrient, gap_kgha, dose_kgha, reason
    """
    blend = []

    # pH - inashughulikiwa kando na N/P/K kwa sababu Lime haihusiani
    # na targets za nutrient, bali na kiwango cha utindikali wa udongo.
    if ph_value < 5.0:
        blend.append({
            "product": "Apply Lime", "nutrient": "pH", "gap_kgha": None,
            "dose_kgha": 1000.0,
            "reason": "pH ni tindikali sana (< 5.0)"
        })
    elif ph_value < 5.5:
        blend.append({
            "product": "Apply Lime", "nutrient": "pH", "gap_kgha": None,
            "dose_kgha": 500.0,
            "reason": "pH ni tindikali (5.0 - 5.5)"
        })

    # Gap halisi ya kila nutrient (kg/ha), kivyake
    gap_n = max(TARGET_N_KGHA - n_kgha, 0)
    gap_p = max(TARGET_P_KGHA - p_kgha, 0)
    gap_k = max(TARGET_K_KGHA - k_kgha, 0)

    nutrient_gaps = [
        ("N", gap_n, "Urea", fertilizer_info["Urea"]["n_pct"]),
        ("P", gap_p, "SSP",  fertilizer_info["SSP"]["p_pct"]),
        ("K", gap_k, "MOP",  fertilizer_info["MOP"]["k_pct"]),
    ]

    for nutrient, gap_kgha, product, pct in nutrient_gaps:
        if gap_kgha > 0:
            dose_kgha = gap_kgha / pct
            dose_kgha = max(MIN_PRODUCT_DOSE_KGHA, min(dose_kgha, MAX_PRODUCT_DOSE_KGHA))
            blend.append({
                "product": product, "nutrient": nutrient,
                "gap_kgha": round(gap_kgha, 1),
                "dose_kgha": round(dose_kgha, 1),
                "reason": f"{nutrient} iko chini ya kiwango kinachohitajika kwa kiasi cha {round(gap_kgha,1)} kg/ha"
            })

    if not blend:
        blend.append({
            "product": "No Fertilizer", "nutrient": "-", "gap_kgha": None,
            "dose_kgha": 0.0,
            "reason": "Udongo tayari una N, P, K na pH vya kutosha kwa mahindi"
        })

    return blend


def get_confidence(input_data, predicted_class_index):
    probabilities = model.predict_proba(input_data)[0]
    confidence_pct = round(probabilities[predicted_class_index] * 100, 1)
    class_names = le_target.inverse_transform(np.arange(len(probabilities)))
    prob_pairs = list(zip(class_names, probabilities))
    prob_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3 = [
        {"name": name, "pct": round(prob * 100, 1)}
        for name, prob in prob_pairs[:3]
        if prob > 0.01
    ]
    return confidence_pct, top_3


# District coordinates + auto-detected soil type kutoka dataset ya 4,998 records
#
# MUHIMU: Model (Random Forest) ilifunzwa kwa districts 7, zote zikiwa na
# class yake halisi (714 records kila moja): Busokelo, Chunya, Kyela,
# Mbarali, Mbeya_Mjini, Mbeya_Vijijini, Rungwe. "model_id" hapa chini
# ZIMETHIBITISHWA moja kwa moja kutoka LabelEncoder(kialfabeti) kwenye
# dataset halisi - Mbeya_Mjini=4, Mbeya_Vijijini=5, Rungwe=6.
# (Awali kulikuwa na bug: Mbeya Vijijini na Rungwe zilikuwa zinatumia
# model_id zisizo sahihi (4 na 5), zikichanganya matokeo ya wilaya hizo
# mbili na yale ya Mbeya Mjini. Imesharekebishwa.)
DISTRICT_COORDS = {
    "0": {"name": "Busokelo",        "lat": -9.00,   "lon": 33.58,   "soil_type": 3, "soil_name": "Sandy Clay Loam", "model_id": 0},
    "1": {"name": "Chunya",          "lat": -8.50,   "lon": 33.42,   "soil_type": 4, "soil_name": "Sandy Loam",      "model_id": 1},
    "2": {"name": "Kyela",           "lat": -9.60,   "lon": 33.85,   "soil_type": 3, "soil_name": "Sandy Clay Loam", "model_id": 2},
    "3": {"name": "Mbarali",         "lat": -8.70,   "lon": 33.72,   "soil_type": 3, "soil_name": "Sandy Clay Loam", "model_id": 3},
    "4": {"name": "Mbeya Vijijini",  "lat": -8.90,   "lon": 33.46,   "soil_type": 3, "soil_name": "Sandy Clay Loam", "model_id": 5},
    "5": {"name": "Rungwe",          "lat": -9.20,   "lon": 33.55,   "soil_type": 3, "soil_name": "Sandy Clay Loam", "model_id": 6},
    "6": {"name": "Mbeya Mjini",     "lat": -8.9094, "lon": 33.4608, "soil_type": 3, "soil_name": "Sandy Clay Loam", "model_id": 4},
}


def get_climate_data(district_id):
    try:
        coords = DISTRICT_COORDS.get(str(district_id))
        if not coords:
            return None
        today = datetime.today().strftime("%Y%m%d")
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,RH2M,PRECTOTCORR",
            "community":  "AG",
            "latitude":   coords["lat"],
            "longitude":  coords["lon"],
            "start":      today,
            "end":        today,
            "format":     "JSON"
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        prop        = data["properties"]["parameter"]
        temperature = round(prop["T2M"][today], 1)
        humidity    = round(prop["RH2M"][today], 1)
        rainfall    = round(prop["PRECTOTCORR"][today], 1)
        if temperature == -999: temperature = 20.0
        if humidity    == -999: humidity    = 70.0
        if rainfall    == -999: rainfall    = 0.0
        return {
            "temperature": temperature,
            "humidity":    humidity,
            "rainfall":    rainfall,
            "source":      "NASA POWER API (real-time)",
            "district":    coords["name"]
        }
    except Exception:
        return {
            "temperature": 20.0,
            "humidity":    70.0,
            "rainfall":    0.0,
            "source":      "Default values (NASA API unavailable)",
            "district":    DISTRICT_COORDS.get(str(district_id), {}).get("name", "Unknown")
        }


# ── AUTH ROUTES ──

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name        = request.form.get("reg_fullname_field", "").strip()
        email            = request.form.get("reg_email_field", "").strip()
        password         = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        if len(full_name) < 2:
            errors.append("Full name must be at least 2 characters.")
        if "@" not in email or "." not in email:
            errors.append("Please enter a valid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")

        if errors:
            return render_template("register.html", errors=errors)

        success = create_user(full_name, email, password)
        if not success:
            return render_template(
                "register.html",
                errors=["An account with this email already exists."]
            )

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", errors=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = verify_user(email, password)
        if user:
            session["user_id"]   = user["id"]
            session["user_name"] = user["full_name"]
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ── MAIN ROUTES ──

@app.route("/")
@login_required
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    errors = validate_inputs(request.form)
    if errors:
        return render_template("error.html", errors=errors), 400

    district  = request.form["district"]
    ph_value  = float(request.form["ph_value"])
    farm_size = float(request.form["farm_size"])

    # Soil type inachukuliwa automatically kutoka dataset kulingana na district
    soil_type = DISTRICT_COORDS.get(str(district), {}).get("soil_type", 3)
    soil_name = DISTRICT_COORDS.get(str(district), {}).get("soil_name", "Sandy Clay Loam")

    # model_id: district id iliyotumika WAKATI wa kufunza model (training).
    # Kwa districts za asili (0-5) hii ni sawa na district id yenyewe.
    # Kwa districts mpya zisizokuwepo kwenye training data (mfano "6" =
    # Mbeya Urban), inatumia model_id ya district iliyo karibu kijiografia
    # ili model itoe utabiri unaotegemewa.
    model_district_id = DISTRICT_COORDS.get(str(district), {}).get("model_id", int(district))

    n_mgkg = float(request.form["n_value"])
    p_mgkg = float(request.form["p_value"])
    k_mgkg = float(request.form["k_value"])

    n_value = n_mgkg * 2
    p_value = p_mgkg * 2
    k_value = k_mgkg * 2

    # Climate data inafetchwa automatically kutoka NASA POWER
    climate     = get_climate_data(district)
    temperature = climate["temperature"]
    humidity    = climate["humidity"]
    rainfall    = climate["rainfall"]

    npk_total = n_value + p_value + k_value
    np_ratio  = n_value / (p_value + 1)

    input_data = np.array([[
        model_district_id, soil_type,
        n_value, p_value, k_value,
        temperature, humidity, ph_value,
        rainfall, npk_total, np_ratio
    ]])

    prediction = model.predict(input_data)[0]
    result     = le_target.inverse_transform([prediction])[0]

    confidence_pct, top_3 = get_confidence(input_data, prediction)

    info = fertilizer_info.get(result, default_info)

    dose_kgha     = calculate_dose(result, n_value, p_value, k_value)
    total_dose_kg = round(dose_kgha * farm_size, 1) if dose_kgha else None

    # Precision Custom Blend (alternative kwa blanket recommendation)
    custom_blend = calculate_custom_blend(ph_value, n_value, p_value, k_value)
    for item in custom_blend:
        item["total_kg"] = round(item["dose_kgha"] * farm_size, 1) if item["dose_kgha"] else 0.0
        item["when"] = fertilizer_info.get(item["product"], default_info)["when"]

    save_prediction({
        "user_id":        session.get("user_id"),
        "user_name":      session.get("user_name"),
        "district":       DISTRICT_COORDS.get(str(district), {}).get("name", district),
        "soil_name":      soil_name,
        "n_value":        n_mgkg,
        "p_value":        p_mgkg,
        "k_value":        k_mgkg,
        "ph_value":       ph_value,
        "temperature":    temperature,
        "humidity":       humidity,
        "rainfall":       rainfall,
        "recommendation": result,
        "dose_kgha":      dose_kgha,
        "farm_size":      farm_size,
        "total_dose_kg":  total_dose_kg,
        "confidence":     confidence_pct,
        "climate_source": climate["source"]
    })

    return render_template("result.html",
                           recommendation=result,
                           info=info,
                           climate=climate,
                           dose_kgha=dose_kgha,
                           farm_size=farm_size,
                           total_dose_kg=total_dose_kg,
                           confidence=confidence_pct,
                           top_3=top_3,
                           soil_name=soil_name,
                           n_mgkg=n_mgkg,
                           p_mgkg=p_mgkg,
                           k_mgkg=k_mgkg,
                           custom_blend=custom_blend)


@app.route("/history")
@login_required
def history():
    predictions = get_all_predictions(user_id=session.get("user_id"))
    return render_template("history.html", predictions=predictions)


@app.route("/export")
@login_required
def export():
    predictions = get_all_predictions(user_id=session.get("user_id"))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "#", "District", "Soil Type", "N (mg/kg)", "P (mg/kg)", "K (mg/kg)",
        "pH", "Temperature", "Humidity", "Rainfall", "Recommendation",
        "Dose (kg/ha)", "Farm Size (ha)", "Total Dose (kg)",
        "Confidence (%)", "Date"
    ])
    for i, p in enumerate(predictions, 1):
        writer.writerow([
            i,
            p["district"],
            p.get("soil_name", "-"),
            p["n_value"],
            p["p_value"],
            p["k_value"],
            p["ph_value"],
            p.get("temperature", ""),
            p.get("humidity", ""),
            p.get("rainfall", ""),
            p["recommendation"],
            p.get("dose_kgha", "-"),
            p.get("farm_size", "-"),
            p.get("total_dose_kg", "-"),
            p.get("confidence", "-"),
            p.get("created_at", "")
        ])
    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fertilizer_history.csv"}
    )


@app.route("/dashboard")
@login_required
def dashboard():
    user_id         = session.get("user_id")
    counts          = get_recommendation_counts(user_id=user_id)
    district_counts = get_district_counts(user_id=user_id)
    total           = get_total_count(user_id=user_id)
    return render_template("dashboard.html",
                           counts=counts,
                           district_counts=district_counts,
                           total=total)


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


def open_browser():
    chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
    try:
        webbrowser.get(chrome_path).open("http://127.0.0.1:5000/login")
    except webbrowser.Error:
        webbrowser.open_new("http://127.0.0.1:5000/login")


# Hakikisha database na tables ziko tayari KILA WAKATI programu
# inapopakiwa (siyo tu wakati wa "python app.py") - hii inazuia
# hitilafu ya "no such table: users" endapo db inafutwa au app
# inapakiwa kwa njia nyingine (mfano WSGI server).
init_db()
migrate_db()

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False)