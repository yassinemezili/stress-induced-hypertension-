# ============================================================
#   Stress-Induced Hypertension Predictor — FastAPI Wrapper
#   Author  : MEZILI Ahmed Yassine
#   ID      : 38098114
# ============================================================
#
#   WHAT THIS FILE DOES
#   -------------------
#   Wraps the trained Gradient Boosting model in a REST API
#   so any wearable device or app can send patient data and
#   receive a hypertension prediction in real-time.
#
#   HOW TO RUN
#   ----------
#   1. Install FastAPI:
#      pip install fastapi uvicorn
#
#   2. Run the API:
#      uvicorn api:app --reload
#
#   3. Open in browser:
#      http://127.0.0.1:8000/docs   ← interactive test page
#
#   ENDPOINTS
#   ---------
#   GET  /          → health check
#   GET  /model     → model info & feature names
#   POST /predict   → single patient prediction
#   POST /predict/batch → multiple patients at once
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ── Load model ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)
    model         = saved['model']
    feature_names = saved['feature_names']
    print(f"Model loaded successfully. Features: {feature_names}")
except FileNotFoundError:
    raise RuntimeError(
        "best_model.pkl not found. "
        "Run hypertension_predictor.py first to train and save the model."
    )

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "Stress-Induced Hypertension Predictor",
    description = (
        "REST API for predicting hypertension risk from wearable device data. "
        "Built with Gradient Boosting — AUC = 1.000, Recall = 0.98."
    ),
    version     = "1.0.0"
)

# ── Decision threshold ────────────────────────────────────────
# 0.40 chosen over 0.50 because in a medical context,
# missing a hypertension case (False Negative) is more
# dangerous than a false alarm (False Positive).
THRESHOLD = 0.40

# ─────────────────────────────────────────────────────────────
# INPUT SCHEMA
# ─────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    # Raw features
    Age             : float = Field(..., ge=18, le=100, example=45,  description="Patient age in years")
    Salt_Intake     : float = Field(..., ge=0,  le=20,  example=9.5, description="Daily salt intake (grams)")
    Stress_Score    : float = Field(..., ge=0,  le=10,  example=8.0, description="Stress score from wearable (0-10)")
    BP_History      : int   = Field(..., ge=0,  le=2,   example=1,   description="0=Normal, 1=Prehypertension, 2=Hypertension")
    Sleep_Duration  : float = Field(..., ge=0,  le=15,  example=5.2, description="Average sleep duration (hours)")
    BMI             : float = Field(..., ge=10, le=60,  example=27.3,description="Body Mass Index")
    Family_History  : int   = Field(..., ge=0,  le=1,   example=1,   description="Family history of hypertension (0=No, 1=Yes)")
    Exercise_Level  : int   = Field(..., ge=0,  le=2,   example=0,   description="0=Low, 1=Moderate, 2=High")
    Smoking_Status  : int   = Field(..., ge=0,  le=1,   example=0,   description="0=Non-Smoker, 1=Smoker")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45, "Salt_Intake": 9.5, "Stress_Score": 8.0,
                "BP_History": 1, "Sleep_Duration": 5.2, "BMI": 27.3,
                "Family_History": 1, "Exercise_Level": 0, "Smoking_Status": 0
            }
        }

# ─────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    hypertension        : bool
    probability         : float
    risk_level          : str
    threshold_used      : float
    engineered_features : dict

class BatchResult(BaseModel):
    predictions : List[PredictionResult]
    total       : int
    high_risk   : int

# ─────────────────────────────────────────────────────────────
# HELPER — feature engineering (must match training pipeline)
# ─────────────────────────────────────────────────────────────
def engineer_features(data: PatientData) -> pd.DataFrame:
    """
    Applies the same feature engineering as the training pipeline.
    Must stay in sync with hypertension_predictor.py Step 3.
    """
    row = {
        'Age'            : data.Age,
        'Salt_Intake'    : data.Salt_Intake,
        'Stress_Score'   : data.Stress_Score,
        'BP_History'     : data.BP_History,
        'Sleep_Duration' : data.Sleep_Duration,
        'BMI'            : data.BMI,
        'Family_History' : data.Family_History,
        'Exercise_Level' : data.Exercise_Level,
        'Smoking_Status' : data.Smoking_Status,
        # Engineered features
        'Stress_Sleep_Ratio'   : data.Stress_Score / (data.Sleep_Duration + 0.1),
        'Metabolic_Risk'       : data.BMI * data.Salt_Intake / 10,
        'Lifestyle_Risk_Score' : (data.Smoking_Status * 2
                                  + (2 - data.Exercise_Level)
                                  + int(data.Sleep_Duration < 6)),
        'Age_Stress'           : data.Age * data.Stress_Score / 100,
    }
    return pd.DataFrame([row])[feature_names]

def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Moderate Risk"
    elif probability < 0.80:
        return "High Risk"
    else:
        return "Very High Risk"

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Check if the API is running."""
    return {
        "status"  : "online",
        "model"   : "Gradient Boosting",
        "version" : "1.0.0"
    }


@app.get("/model", tags=["Info"])
def model_info():
    """Returns model details, feature names, and threshold."""
    return {
        "model_type"     : "Gradient Boosting Classifier",
        "n_estimators"   : 200,
        "auc_score"      : 1.000,
        "recall"         : 0.980,
        "threshold"      : THRESHOLD,
        "feature_names"  : feature_names,
        "total_features" : len(feature_names)
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(patient: PatientData):
    """
    Predict hypertension risk for a single patient.

    Send raw wearable + clinical data and receive:
    - hypertension: True/False
    - probability: risk score (0.0 to 1.0)
    - risk_level: Low / Moderate / High / Very High
    """
    try:
        df_input    = engineer_features(patient)
        probability = float(model.predict_proba(df_input)[0, 1])
        prediction  = probability >= THRESHOLD

        # Return engineered features so caller can inspect them
        eng = {
            'Stress_Sleep_Ratio'   : round(float(df_input['Stress_Sleep_Ratio'].iloc[0]), 3),
            'Metabolic_Risk'       : round(float(df_input['Metabolic_Risk'].iloc[0]), 3),
            'Lifestyle_Risk_Score' : round(float(df_input['Lifestyle_Risk_Score'].iloc[0]), 3),
            'Age_Stress'           : round(float(df_input['Age_Stress'].iloc[0]), 3),
        }

        return PredictionResult(
            hypertension        = bool(prediction),
            probability         = round(probability, 3),
            risk_level          = get_risk_level(probability),
            threshold_used      = THRESHOLD,
            engineered_features = eng
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(patients: List[PatientData]):
    """
    Predict hypertension risk for multiple patients at once.
    Useful for processing data from a group of wearable devices.
    """
    if len(patients) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 100 patients per request."
        )

    predictions = []
    high_risk   = 0

    for patient in patients:
        try:
            df_input    = engineer_features(patient)
            probability = float(model.predict_proba(df_input)[0, 1])
            prediction  = probability >= THRESHOLD

            if prediction:
                high_risk += 1

            eng = {
                'Stress_Sleep_Ratio'   : round(float(df_input['Stress_Sleep_Ratio'].iloc[0]), 3),
                'Metabolic_Risk'       : round(float(df_input['Metabolic_Risk'].iloc[0]), 3),
                'Lifestyle_Risk_Score' : round(float(df_input['Lifestyle_Risk_Score'].iloc[0]), 3),
                'Age_Stress'           : round(float(df_input['Age_Stress'].iloc[0]), 3),
            }

            predictions.append(PredictionResult(
                hypertension        = bool(prediction),
                probability         = round(probability, 3),
                risk_level          = get_risk_level(probability),
                threshold_used      = THRESHOLD,
                engineered_features = eng
            ))

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error on patient: {str(e)}")

    return BatchResult(
        predictions = predictions,
        total       = len(predictions),
        high_risk   = high_risk
    )
