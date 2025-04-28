import re
import fitz  # PyMuPDF
import joblib
import numpy as np

# Load trained components
model = joblib.load("model_xgboost.joblib")
encoders = joblib.load("encoders.joblib")
label_encoders = joblib.load("label_encoders.joblib")
mlb_family = joblib.load("mlb_family.joblib")
mlb_symptoms = joblib.load("mlb_symptoms.joblib")

# Extract a single value or list from PDF text
def extract_value(text, label, default=None, as_list=False):
    match = re.search(f"{label}:\\s*(.+)", text, re.IGNORECASE)
    if not match:
        return default
    return (
        [item.strip().lower() for item in match.group(1).split(",")] if as_list
        else match.group(1).strip().lower()
    )

# Extract all health fields from PDF
def extract_health_data_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = " ".join([page.get_text() for page in doc])

    return {
        "age": int(extract_value(text, "Age", 30)),
        "gender": extract_value(text, "Gender", "male"),
        "smoking": extract_value(text, "Smoking", "no"),
        "alcohol": extract_value(text, "Alcohol", "no"),
        "exercise": extract_value(text, "Exercise", "sometimes"),
        "sleep": extract_value(text, "Sleep Duration", "5-7"),
        "diet": extract_value(text, "Diet", "average"),
        "weight": float(extract_value(text, "Weight", 70)),
        "stress": extract_value(text, "Stress Level", "moderate"),
        "bloodPressure": float(extract_value(text, "Blood Pressure", 120)),
        "sugarLevel": float(extract_value(text, "Blood Sugar", 100)),
        "cholesterol": float(extract_value(text, "Cholesterol", 180)),
        "mentalHealth": extract_value(text, "Mental Health", "no"),
        "activityLevel": extract_value(text, "Physical Activity", "moderate"),
        "familyHistory": extract_value(text, "Family History", ["heart"], as_list=True),
        "symptoms": extract_value(text, "Symptoms", ["tired"], as_list=True),
    }

# Generate detailed recommendations based on risk levels
def generate_recommendations(predictions):
    full_recs = {
        "heart": {
            "High": [
                "Engage in aerobic exercise 5x/week.",
                "Avoid smoking and secondhand smoke.",
                "Limit sodium and saturated fats.",
                "Monitor blood pressure weekly.",
                "Manage stress actively (yoga, therapy).",
                "Visit a cardiologist for screening."
            ],
            "Moderate": [
                "Walk 30 mins/day, 3-4 days/week.",
                "Reduce salty snacks and fried items.",
                "Monthly BP checks advised.",
                "Switch to olive/sunflower oil."
            ],
            "Low": [
                "Continue regular cardio exercise.",
                "Annual ECG and cholesterol check."
            ]
        },
        "diabetes": {
            "High": [
                "Eliminate sugary drinks and desserts.",
                "Adopt a low-GI, fiber-rich diet.",
                "Check fasting sugar weekly.",
                "Sleep consistently 7–8 hours.",
                "See an endocrinologist for control.",
                "Use whole grains and legumes daily."
            ],
            "Moderate": [
                "Avoid polished rice and refined flour.",
                "Walk after meals to lower sugar spikes.",
                "Check HbA1c every 3 months.",
                "Control portion size and carbs."
            ],
            "Low": [
                "Limit sweets and soda occasionally.",
                "Stay active to maintain glucose levels."
            ]
        },
        "mental": {
            "High": [
                "Consult a licensed therapist.",
                "Try 15 mins of guided meditation daily.",
                "Avoid screen time before bed.",
                "Maintain strong social support.",
                "Sleep 7–8 hours undisturbed.",
                "Join stress-relief group activities."
            ],
            "Moderate": [
                "Avoid doomscrolling and news overload.",
                "Keep a journal of your thoughts.",
                "Exercise at least 3x/week.",
                "Practice gratitude regularly."
            ],
            "Low": [
                "Continue healthy routines and rest.",
                "Stay mentally engaged through hobbies."
            ]
        },
        "obesity": {
            "High": [
                "Follow a calorie-deficit meal plan.",
                "Eliminate refined carbs and soda.",
                "Do strength + cardio weekly.",
                "Track meals using fitness apps.",
                "Get a thyroid profile tested.",
                "Avoid stress-eating triggers."
            ],
            "Moderate": [
                "Focus on whole foods and proteins.",
                "Stay hydrated (2–3L/day).",
                "Include 30 mins of exercise daily.",
                "Limit high-fat street foods."
            ],
            "Low": [
                "Maintain diet and active routine.",
                "Avoid night snacks and binge eating."
            ]
        }
    }

    result = {}
    for key in ["heart", "diabetes", "mental", "obesity"]:
        level = predictions[f"{key}Risk"]
        result[f"{key}Advice"] = full_recs[key][level]
    return result

# Predict from extracted PDF fields
def predict_from_extracted_fields(data):
    features = [
        "age", "gender", "smoking", "alcohol", "exercise", "sleep",
        "diet", "weight", "stress", "bloodPressure", "sugarLevel",
        "cholesterol", "mentalHealth", "activityLevel"
    ]

    x = []
    for col in features:
        val = data[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        x.append(val)

    fam_vec = mlb_family.transform([data["familyHistory"]])
    sym_vec = mlb_symptoms.transform([data["symptoms"]])
    final_input = np.hstack([x, fam_vec[0], sym_vec[0]]).reshape(1, -1)

    y_pred = model.predict(final_input)[0]
    risks = {}
    for i, key in enumerate(["heartRisk", "diabetesRisk", "mentalRisk", "obesityRisk"]):
        risks[key] = label_encoders[key].inverse_transform([y_pred[i]])[0]

    recs = generate_recommendations(risks)

    return {
        "data": {
            **data,
            "familyHistory": ",".join(data["familyHistory"]),
            "symptoms": ",".join(data["symptoms"])
        },
        "result": {
            **risks,
            **recs
        }
    }
