from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
import joblib
import numpy as np
import shutil
from upload import extract_health_data_from_pdf, predict_from_extracted_fields

# Load models and encoders
model = joblib.load("model_xgboost.joblib")
encoders = joblib.load("encoders.joblib")
label_encoders = joblib.load("label_encoders.joblib")
mlb_family = joblib.load("mlb_family.joblib")
mlb_symptoms = joblib.load("mlb_symptoms.joblib")

# MongoDB Setup
mongo_uri = "mongodb+srv://sabariS:Q5dQQb31dzzwZ2mL@cluster0.tdgspkd.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client["healthrisk"]
collection = db["surveys"]

# Initialize app
app = FastAPI()

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Pydantic input schema for /api/survey
class SurveyData(BaseModel):
    username: str
    age: int
    gender: str
    smoking: str
    alcohol: str
    exercise: str
    sleep: str
    diet: str
    weight: float
    stress: str
    familyHistory: str
    symptoms: str
    bloodPressure: float
    sugarLevel: float
    cholesterol: float
    mentalHealth: str
    activityLevel: str

# Generate Recommendations
def generate_recommendations(predictions):
    full_recs = {
        "heart": {
            "High": [
                "Engage in at least 30 minutes of aerobic exercise 5 days a week.",
                "Avoid smoking and secondhand smoke completely.",
                "Limit sodium and saturated fats; prefer a DASH diet.",
                "Monitor blood pressure weekly and maintain a log.",
                "Reduce stress through yoga, meditation, or counseling.",
                "Schedule a cardiologist appointment for detailed heart screening."
            ],
            "Moderate": [
                "Start walking 20–30 minutes a day, 3–4 times a week.",
                "Cut down on processed foods and salty snacks.",
                "Get your blood pressure checked monthly.",
                "Use heart-friendly oils like olive or sunflower oil."
            ],
            "Low": [
                "Continue regular physical activity and balanced diet.",
                "Have an annual general health checkup including ECG."
            ]
        },
        "diabetes": {
            "High": [
                "Avoid sugary foods and beverages entirely.",
                "Follow a high-fiber, low-GI diet with vegetables and whole grains.",
                "Check fasting and postprandial blood sugar regularly.",
                "Maintain a stable sleep schedule and reduce late-night eating.",
                "Consult a diabetologist for medication or management plan.",
                "Incorporate cinnamon, bitter melon, or fenugreek into diet."
            ],
            "Moderate": [
                "Avoid white rice and refined flours.",
                "Limit fruit juices and opt for whole fruits instead.",
                "Engage in moderate physical activity daily.",
                "Monitor HbA1c every 3 months."
            ],
            "Low": [
                "Maintain a healthy diet and avoid unnecessary sugar.",
                "Keep your weight and waist circumference in check."
            ]
        },
        "mental": {
            "High": [
                "Seek help from a mental health professional.",
                "Practice mindfulness for 15 minutes daily.",
                "Ensure consistent 7–8 hour sleep.",
                "Avoid caffeine and alcohol during stress.",
                "Stay socially connected and avoid isolation.",
                "Journal thoughts and express emotions regularly."
            ],
            "Moderate": [
                "Use breathing exercises daily.",
                "Limit screen time and late-night scrolling.",
                "Talk with friends or therapist.",
                "Maintain consistent sleep/wake times."
            ],
            "Low": [
                "Continue social interaction and stress-free routines.",
                "Maintain good sleep hygiene and mental stimulation."
            ]
        },
        "obesity": {
            "High": [
                "Follow a calorie-deficit diet from a dietician.",
                "Avoid sugary drinks and processed carbs.",
                "Do strength + cardio workouts weekly.",
                "Track your meals using apps.",
                "Check for thyroid/hormonal imbalances.",
                "Avoid emotional and binge eating."
            ],
            "Moderate": [
                "Eat fiber-rich and protein-balanced meals.",
                "Drink 2–3L of water daily.",
                "Walk at least 30 mins per day.",
                "Use stairs regularly and avoid long sitting."
            ],
            "Low": [
                "Maintain current physical activity and balanced nutrition.",
                "Avoid fast foods and excess oil."
            ]
        }
    }

    result = {}
    for i, key in enumerate(["heart", "diabetes", "mental", "obesity"]):
        level = predictions[f"{key}Risk"]
        result[f"{key}Advice"] = full_recs[key][level]
    return result

# POST /api/survey — form submission
@app.post("/api/survey")
async def analyze_risk(data: SurveyData):
    try:
        input_dict = data.dict()
        family = input_dict["familyHistory"].split(",")
        symptoms = input_dict["symptoms"].split(",")

        feature_cols = [
            "age", "gender", "smoking", "alcohol", "exercise", "sleep",
            "diet", "weight", "stress", "bloodPressure", "sugarLevel",
            "cholesterol", "mentalHealth", "activityLevel"
        ]

        input_arr = []
        for col in feature_cols:
            val = input_dict[col]
            if col in encoders:
                val = encoders[col].transform([val])[0]
            input_arr.append(val)

        fam_vec = mlb_family.transform([family])
        sym_vec = mlb_symptoms.transform([symptoms])
        final_input = np.hstack([input_arr, fam_vec[0], sym_vec[0]]).reshape(1, -1)

        raw_preds = model.predict(final_input)[0]
        risks = {}
        for i, key in enumerate(["heartRisk", "diabetesRisk", "mentalRisk", "obesityRisk"]):
            risks[key] = label_encoders[key].inverse_transform([raw_preds[i]])[0]

        advice = generate_recommendations(risks)
        result = {"data": input_dict, "result": {**risks, **advice}}

        # Save to MongoDB
        collection.insert_one({
            "username": input_dict["username"],
            "source": "survey",
            "data": input_dict,
            "result": result["result"],
            "timestamp": datetime.utcnow()
        })

        return result

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(e)})

# POST /api/upload — PDF upload
@app.post("/api/upload")
async def upload_report(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF to disk
        file_location = f"temp_uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract data from PDF
        extracted = extract_health_data_from_pdf(file_location)

        # Predict based on extracted values
        result = predict_from_extracted_fields(extracted)

        # Save record to MongoDB
        collection.insert_one({
            "username": extracted.get("username", "Anonymous"),
            "source": "pdf",
            "data": extracted,
            "result": result["result"],  # Make sure this key exists in 'result'
            "timestamp": datetime.utcnow()
        })

        return result

    except Exception as e:
        print("🔥 UPLOAD ERROR:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to process PDF", "detail": str(e)})

# GET /api/history — fetch user report history
@app.get("/api/history")
async def get_user_history():
    try:
        # Fetch all health risk history records from MongoDB
        entries = list(collection.find().sort("timestamp", -1))  # No filter, fetch all documents sorted by timestamp
        for entry in entries:
            entry["_id"] = str(entry["_id"])  # Convert MongoDB ObjectId to string for frontend usage
        return entries  # Return the list of health report entries
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to fetch history", "detail": str(e)})
