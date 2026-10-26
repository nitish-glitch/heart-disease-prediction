"""
validation.py
=============
Real-world input validation rules for the CardioScreen AI Heart Health system.
Ensures physiological plausibility and blocks invalid/nonsense inputs (e.g. age 911).
"""

def perform_rigorous_validation(age_val, bp_val, chol_val, hr_val, st_val):
    """
    Validates all numeric inputs against strict medical plausibility boundaries.
    Explicitly catches bugs like age 911 or non-sensical values.
    Returns (is_valid, user_friendly_error_message).
    """
    # 1. Age validation (Critical bug fix)
    if age_val is None or not (1 <= age_val <= 120):
        return False, "Please enter a valid age between 1 and 120."
        
    # 2. Blood Pressure validation
    if bp_val is None or not (60 <= bp_val <= 240):
        return False, "Please enter a realistic resting blood pressure between 60 and 240 mm Hg."
        
    # 3. Cholesterol validation
    if chol_val is None or not (80 <= chol_val <= 600):
        return False, "Please enter a realistic cholesterol level between 80 and 600 mg/dL."
        
    # 4. Maximum Heart Rate validation
    if hr_val is None or not (50 <= hr_val <= 220):
        return False, "Please enter a realistic maximum heart rate between 50 and 220 beats per minute."
        
    # 5. ST Depression validation
    if st_val is None or not (0.0 <= st_val <= 8.0):
        return False, "Please enter an ECG wave shift value between 0.0 and 8.0 mm."
        
    return True, "Valid"
