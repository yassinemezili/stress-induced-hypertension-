# ============================================================
#   HyperGuard — FastAPI Backend
#   Author  : MEZILI Ahmed Yassine | ID: 38098114
#   Run with: python server.py
#   Opens   : http://localhost:8000
# ============================================================

import pickle, uvicorn, webbrowser, threading, os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np

BASE_DIR = Path(__file__).parent
app = FastAPI(title="HyperGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load model ────────────────────────────────────────────────
MODEL, FEATURE_NAMES = None, None
model_path = BASE_DIR / "best_model.pkl"
if model_path.exists():
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    MODEL        = saved["model"]
    FEATURE_NAMES = saved["feature_names"]
    print(f"✅ Model loaded — {len(FEATURE_NAMES)} features")
else:
    print("⚠️  best_model.pkl not found — predictions will return demo data")

# ── Schemas ───────────────────────────────────────────────────
class DayReading(BaseModel):
    sleep: float        # hours
    stress: int         # 1–10
    salt: float         # g/day
    bp_sys: int         # mmHg, 0 if not measured
    bp_dia: int         # mmHg, 0 if not measured

class AssessRequest(BaseModel):
    age:            int
    bmi:            float
    bp_history:     int     # 0/1
    family_history: int     # 0/1
    smoking:        int     # 0/1
    exercise:       int     # 0/1/2
    days:           List[DayReading]  # exactly 7

# ── Feature engineering ───────────────────────────────────────
def build_features(day: DayReading, p: AssessRequest) -> dict:
    return {
        "Age":                 p.age,
        "Salt_Intake":         day.salt,
        "Stress_Score":        day.stress,
        "BP_History":          p.bp_history,
        "Sleep_Duration":      day.sleep,
        "BMI":                 p.bmi,
        "Family_History":      p.family_history,
        "Exercise_Level":      p.exercise,
        "Smoking_Status":      p.smoking,
        "Stress_Sleep_Ratio":  day.stress / (day.sleep + 0.1),
        "Metabolic_Risk":      p.bmi * day.salt / 10,
        "Lifestyle_Risk_Score":p.smoking * 2 + (2 - p.exercise) + int(day.sleep < 6),
        "Age_Stress":          p.age * day.stress / 100,
    }

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/status")
async def status():
    return {"model_loaded": MODEL is not None, "features": FEATURE_NAMES}

@app.post("/predict")
async def predict(req: AssessRequest):
    if len(req.days) != 7:
        return JSONResponse({"error": "Exactly 7 days required"}, status_code=400)

    daily_probs = []
    import pandas as pd

    for day in req.days:
        feat  = build_features(day, req)
        if MODEL:
            df    = pd.DataFrame([feat])[FEATURE_NAMES]
            prob  = float(MODEL.predict_proba(df)[0, 1])
        else:
            # demo fallback
            prob = min(1.0, (day.stress / 10 * 0.4 + (1 - day.sleep / 10) * 0.3 +
                             day.salt / 20 * 0.3 + req.bp_history * 0.2 +
                             req.family_history * 0.1))
        daily_probs.append(round(prob, 4))

    # Average-based final prediction
    all_feats = [build_features(d, req) for d in req.days]
    if MODEL:
        avg_df     = pd.DataFrame(all_feats)[FEATURE_NAMES].mean().to_frame().T
        final_prob = float(MODEL.predict_proba(avg_df)[0, 1])
    else:
        final_prob = float(np.mean(daily_probs))

    final_prob = round(final_prob, 4)

    # BP stats
    bp_days    = [d for d in req.days if d.bp_sys > 0]
    avg_sys    = round(np.mean([d.bp_sys for d in bp_days]), 1) if bp_days else None
    avg_dia    = round(np.mean([d.bp_dia for d in bp_days]), 1) if bp_days else None

    # Recommendations
    days       = req.days
    hi_stress  = sum(1 for d in days if d.stress >= 7)
    low_sleep  = sum(1 for d in days if d.sleep < 6)
    hi_salt    = sum(1 for d in days if d.salt > 8)
    recs       = []
    if hi_stress >= 3:
        recs.append(f"{hi_stress}/7 days with high stress — consider mindfulness or workload reduction.")
    if low_sleep >= 2:
        recs.append(f"{low_sleep}/7 nights under 6h sleep — aim for 7–9 hours consistently.")
    if hi_salt >= 3:
        recs.append(f"{hi_salt}/7 days above 8g salt — WHO recommends under 5g/day.")
    if req.smoking:
        recs.append("Smoking directly raises hypertension risk — consider cessation programs.")
    if req.exercise == 0:
        recs.append("No exercise detected — even 30 min walking 3×/week lowers BP significantly.")
    if req.bmi > 27:
        recs.append(f"BMI {req.bmi} is above ideal — even a 5% weight reduction helps lower BP.")
    if not recs:
        recs.append("Your weekly profile looks healthy — keep maintaining current habits.")

    return {
        "final_prob":  final_prob,
        "final_pred":  int(final_prob >= 0.40),
        "daily_probs": daily_probs,
        "avg_stress":  round(np.mean([d.stress for d in days]), 1),
        "avg_sleep":   round(np.mean([d.sleep  for d in days]), 1),
        "avg_salt":    round(np.mean([d.salt   for d in days]), 1),
        "avg_sys":     avg_sys,
        "avg_dia":     avg_dia,
        "recommendations": recs,
    }

# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    def open_browser():
        import time; time.sleep(1.2)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    print("\n🩺 HyperGuard running at http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
