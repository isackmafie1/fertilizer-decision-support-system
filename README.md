# Fertilizer Decision Support System (Mbeya, Tanzania)

A bilingual (English/Swahili) Flask web application that recommends fertilizer type, dosage and a confidence score for maize farmers and extension officers across 7 districts of Mbeya Region.

**How it works**
- A Random Forest model trained on 4,998 soil and climate records classifies the best fertilizer (Apply Lime, DAP, MOP, NPK, No Fertilizer, SSP, Urea). Test accuracy is about 94%.
- A "Precision Fertilizer Plan" calculates the dose from the gap between the soil's nutrient levels and crop needs.
- Live climate data comes from the NASA POWER API, with an offline fallback if the API is unavailable.
- User accounts, login and prediction history are stored in SQLite.

**Tech stack:** Python, Flask, scikit-learn, SQLite, NASA POWER API, Bootstrap.

**Run locally**
```
cd backend
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000/login, register an account, and enter soil details on the home page. Optionally set `DSS_SECRET_KEY` to a long random string.

*Capstone project, Bachelor of Engineering in Data Science, Mbeya University of Science and Technology. Detailed notes in Swahili follow below; technical documentation is in DOCUMENTATION.md.*

---

