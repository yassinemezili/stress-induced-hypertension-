# ============================================================
#   Stress-Induced Hypertension Predictor — Core Logic
#   Author  : MEZILI Ahmed Yassine | ID: 38098114
#   (Streamlit UI removed - ML functions preserved)
# ============================================================

import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def load_model():
    path = BASE_DIR / 'best_model.pkl'
    if not path.exists():
        return None, None
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['feature_names']

def get_risk(prob):
    if prob < 0.25:   return "Low Risk",      "badge-low",      "✅", "#10b981"
    elif prob < 0.50: return "Moderate Risk",  "badge-moderate", "⚠️", "#f59e0b"
    elif prob < 0.75: return "High Risk",      "badge-high",     "🔴", "#ef4444"
    else:             return "Critical Risk",  "badge-critical", "🚨", "#7f1d1d"

STRESS_LABELS = {
    1: "1 — Very calm",  2: "2 — Relaxed",     3: "3 — Mild tension",
    4: "4 — Noticeable", 5: "5 — Moderate",    6: "6 — Somewhat stressed",
    7: "7 — High",       8: "8 — Very high",   9: "9 — Severe",
    10: "10 — Extreme"
}

def compute_features(row, profile):
    """Compute all 13 model features from one day's readings + profile."""
    age      = profile['age']
    smoking  = profile['smoking']
    exercise = profile['exercise']
    
    return {
        'Age':                 age,
        'Salt_Intake':         row['salt'],
        'Stress_Score':        row['stress'],
        'BP_History':          profile['bp_hist'],
        'Sleep_Duration':      row['sleep'],
        'BMI':                 profile['bmi'],
        'Family_History':      profile['fam_hist'],
        'Exercise_Level':      exercise,
        'Smoking_Status':      smoking,
        'Stress_Sleep_Ratio':  row['stress'] / (row['sleep'] + 0.1),
        'Metabolic_Risk':      profile['bmi'] * row['salt'] / 10,
        'Lifestyle_Risk_Score': smoking*2 + (2-exercise) + int(row['sleep'] < 6),
        'Age_Stress':          age * row['stress'] / 100,
    }

def plot_exists(fname):
    return (BASE_DIR / fname).exists()

# ══════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════

def predict_patient_risk(profile, daily_readings):
    """
    Make hypertension risk prediction for a patient.
    
    Args:
        profile: dict with keys {age, bmi, bp_hist, fam_hist, smoking, exercise}
        daily_readings: list of 7 dicts with keys {sleep, stress, salt}
    
    Returns:
        dict with prediction results
    """
    model, feature_names = load_model()
    if model is None:
        return {"error": "Model not found"}
    
    # Validate input
    if len(daily_readings) != 7:
        return {"error": "Exactly 7 days of readings required"}
    
    daily_probs = []
    all_rows = []
    
    for day in daily_readings:
        feat = compute_features(day, profile)
        all_rows.append(feat)
        
        # Per-day probability
        day_df = pd.DataFrame([feat])[feature_names]
        day_prob = model.predict_proba(day_df)[0, 1]
        daily_probs.append(day_prob)
    
    # Average across the week
    week_df = pd.DataFrame(all_rows)[feature_names]
    avg_df = week_df.mean().to_frame().T
    
    # Final prediction
    final_prob = model.predict_proba(avg_df)[0, 1]
    final_pred = int(final_prob >= 0.40)
    
    risk_label, badge_class, risk_icon, risk_color = get_risk(final_prob)
    
    # Calculate statistics
    avg_stress = np.mean([d['stress'] for d in daily_readings])
    avg_sleep  = np.mean([d['sleep'] for d in daily_readings])
    avg_salt   = np.mean([d['salt'] for d in daily_readings])
    
    # Recommendations
    recs = []
    hi_stress_days = sum(1 for d in daily_readings if d['stress'] >= 7)
    low_sleep_days = sum(1 for d in daily_readings if d['sleep'] < 6)
    hi_salt_days   = sum(1 for d in daily_readings if d['salt'] > 8)
    
    if hi_stress_days >= 3:
        recs.append(f"Stress management: {hi_stress_days}/7 days with high stress (≥7)")
    if low_sleep_days >= 2:
        recs.append(f"Sleep improvement: {low_sleep_days}/7 days with <6 hours sleep")
    if hi_salt_days >= 3:
        recs.append(f"Reduce salt: {hi_salt_days}/7 days above 8g/day")
    if profile['smoking'] == 1:
        recs.append("Quit smoking: Direct risk factor for hypertension")
    if profile['exercise'] == 0:
        recs.append("Start exercising: 30min walking 3x/week significantly lowers BP")
    if profile['bmi'] > 27:
        recs.append(f"Weight management: BMI {profile['bmi']} is above ideal")
    if not recs:
        recs.append("Your weekly profile looks healthy! Keep maintaining current habits.")
    
    return {
        'final_prob': final_prob,
        'final_pred': final_pred,
        'risk_label': risk_label,
        'risk_icon': risk_icon,
        'daily_probs': daily_probs,
        'avg_stress': avg_stress,
        'avg_sleep': avg_sleep,
        'avg_salt': avg_salt,
        'recommendations': recs,
    }

# ══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Sample patient
    profile = {
        'age': 45,
        'bmi': 27.3,
        'bp_hist': 1,
        'fam_hist': 1,
        'smoking': 0,
        'exercise': 0,
    }
    
    # 7 days of readings
    daily_readings = [
        {'sleep': 5.2, 'stress': 8, 'salt': 9.5},
        {'sleep': 5.5, 'stress': 7, 'salt': 8.0},
        {'sleep': 6.0, 'stress': 9, 'salt': 10.0},
        {'sleep': 5.1, 'stress': 8, 'salt': 9.0},
        {'sleep': 4.8, 'stress': 8, 'salt': 8.5},
        {'sleep': 5.3, 'stress': 7, 'salt': 9.0},
        {'sleep': 5.0, 'stress': 8, 'salt': 9.5},
    ]
    
    result = predict_patient_risk(profile, daily_readings)
    
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Risk Level     : {result['risk_label']} {result['risk_icon']}")
    print(f"Probability    : {result['final_prob']:.1%}")
    print(f"Prediction     : {'HYPERTENSION RISK DETECTED' if result['final_pred'] else 'NO SIGNIFICANT RISK'}")
    print(f"Avg Stress     : {result['avg_stress']:.1f}/10")
    print(f"Avg Sleep      : {result['avg_sleep']:.1f} hours")
    print(f"Avg Salt       : {result['avg_salt']:.1f} g/day")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")
