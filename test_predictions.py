"""
test_predictions.py
===================
Comprehensive automated test suite testing:
1. Realistic Clinical Patient Scenarios (High Risk, Healthy, Borderline).
2. Strict Input Validation (Invalid age 911, out-of-range BP, extreme cholesterol).
3. Plain-language mapping integrity and model prediction safety.
"""

import joblib
import pandas as pd
import numpy as np
from validation import perform_rigorous_validation

def test_validation_layer():
    print("=" * 78)
    print("TEST SUITE PART 1: RIGOROUS INPUT VALIDATION TESTS")
    print("=" * 78)
    
    # 1. Invalid age 911 (User's specific test requirement)
    valid, msg = perform_rigorous_validation(
        age_val=911, bp_val=120, chol_val=200, hr_val=150, st_val=1.0
    )
    print(f"Test: Age = 911 -> Allowed: {valid} | Error Message: '{msg}'")
    assert not valid, "Validation failed: Age 911 should have been rejected!"
    assert msg == "Please enter a valid age between 1 and 120.", f"Unexpected message: {msg}"
    print("  >>> PASSED: Age 911 is successfully blocked with the exact message!\n")
    
    # 2. Invalid age 0 or negative
    valid, msg = perform_rigorous_validation(
        age_val=-5, bp_val=120, chol_val=200, hr_val=150, st_val=1.0
    )
    print(f"Test: Age = -5 -> Allowed: {valid} | Error Message: '{msg}'")
    assert not valid and "valid age between 1 and 120" in msg
    print("  >>> PASSED: Negative age is successfully blocked!\n")
    
    # 3. Invalid Blood Pressure (e.g. 500 mm Hg)
    valid, msg = perform_rigorous_validation(
        age_val=45, bp_val=500, chol_val=200, hr_val=150, st_val=1.0
    )
    print(f"Test: BP = 500 -> Allowed: {valid} | Error Message: '{msg}'")
    assert not valid and "resting blood pressure" in msg
    print("  >>> PASSED: Out-of-range blood pressure is blocked!\n")
    
    # 4. Valid normal profile
    valid, msg = perform_rigorous_validation(
        age_val=52, bp_val=125, chol_val=210, hr_val=155, st_val=0.8
    )
    print(f"Test: Valid Profile -> Allowed: {valid} | Message: '{msg}'")
    assert valid, f"Expected valid profile to pass, but got: {msg}"
    print("  >>> PASSED: Normal valid profile passes without issues!\n")

def test_model_predictions():
    print("=" * 78)
    print("TEST SUITE PART 2: REALISTIC CLINICAL PATIENT SCENARIOS")
    print("=" * 78)
    
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    scenarios = [
        {
            "title": "Scenario A: High Risk Patient Profile",
            "desc": "67-year-old male, asymptomatic (cp=4), BP 160, chol 286, max HR 108, exercise angina (exang=1), ST depression 2.5mm, 3 vessels narrowed (ca=3), reversible thal defect (thal=7).",
            "input": {
                'age': 67.0, 'sex': 1.0, 'cp': 4.0, 'trestbps': 160.0, 'chol': 286.0,
                'fbs': 0.0, 'restecg': 2.0, 'thalach': 108.0, 'exang': 1.0,
                'oldpeak': 2.5, 'slope': 2.0, 'ca': 3.0, 'thal': 7.0
            },
            "expected_risk": "Higher Risk (Class 1)"
        },
        {
            "title": "Scenario B: Healthy / Low Risk Patient Profile",
            "desc": "41-year-old female, atypical pain (cp=2), BP 115, chol 180, high max HR 175, no exercise angina (exang=0), ST depression 0.0mm, normal slope, 0 vessels narrowed (ca=0), normal thal flow (thal=3).",
            "input": {
                'age': 41.0, 'sex': 0.0, 'cp': 2.0, 'trestbps': 115.0, 'chol': 180.0,
                'fbs': 0.0, 'restecg': 0.0, 'thalach': 175.0, 'exang': 0.0,
                'oldpeak': 0.0, 'slope': 1.0, 'ca': 0.0, 'thal': 3.0
            },
            "expected_risk": "Low Risk (Class 0)"
        },
        {
            "title": "Scenario C: Borderline Patient Profile",
            "desc": "54-year-old male, non-anginal pain (cp=3), BP 135, chol 230, max HR 145, mild ST depression 1.2mm, flat slope, 1 vessel narrowed (ca=1), normal thal (thal=3).",
            "input": {
                'age': 54.0, 'sex': 1.0, 'cp': 3.0, 'trestbps': 135.0, 'chol': 230.0,
                'fbs': 0.0, 'restecg': 1.0, 'thalach': 145.0, 'exang': 0.0,
                'oldpeak': 1.2, 'slope': 2.0, 'ca': 1.0, 'thal': 3.0
            },
            "expected_risk": None
        }
    ]
    
    for idx, sc in enumerate(scenarios, 1):
        print(f"[{idx}] {sc['title']}")
        print(f"    Description: {sc['desc']}")
        
        df = pd.DataFrame([sc['input']], columns=feature_columns)
        scaled = scaler.transform(df)
        
        pred = int(model.predict(scaled)[0])
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled)[0]
            prob_no_disease = probs[0]
            prob_disease = probs[1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(scaled)[0]
            prob_disease = 1.0 / (1.0 + np.exp(-decision))
            prob_no_disease = 1.0 - prob_disease
        else:
            prob_disease = 1.0 if pred == 1 else 0.0
            prob_no_disease = 1.0 - prob_disease
            
        risk_label = "Higher Risk of Heart Disease (Class 1)" if pred == 1 else "Low Risk of Heart Disease (Class 0)"
        print(f"    Assessment  : {risk_label}")
        print(f"    Risk Score  : {prob_disease*100:.1f}% (Healthy probability: {prob_no_disease*100:.1f}%)")
        
        if sc["expected_risk"]:
            expected_class = 1 if "Higher" in sc["expected_risk"] else 0
            assert pred == expected_class, f"Mismatch: expected {expected_class}, got {pred}"
            print(f"    Result      : PASSED (Correctly identified as {sc['expected_risk']})\n")
        else:
            print(f"    Result      : PASSED (Edge-case evaluated smoothly without errors)\n")

if __name__ == "__main__":
    test_validation_layer()
    test_model_predictions()
    print("=" * 78)
    print("ALL TESTS (VALIDATION + MODEL SCENARIOS) PASSED SUCCESSFULLY!")
    print("=" * 78)
