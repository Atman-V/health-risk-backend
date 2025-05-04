from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
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
users_collection = db["users"]

# Initialize FastAPI app
app = FastAPI()

# CORS setup to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing and JWT context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key"  # Make sure to store securely in environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2PasswordBearer to handle authorization headers
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# Custom validation error handler for request validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Helper function to create JWT access tokens
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Pydantic input schemas for registration, login, and survey data
class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

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
    familyHistory: list[str]  # Correcting to list of strings
    symptoms: list[str]       # Correcting to list of strings
    bloodPressure: float
    sugarLevel: float
    cholesterol: float
    mentalHealth: str
    activityLevel: str


# Generate Recommendations for health risks
# Generate Recommendations for health risks based on specific survey combinations
def generate_dynamic_recommendations(predictions, survey_data):
    recs = []

    # Heart Risk Recommendations based on survey combinations
    if predictions["heartRisk"] == "High":
        if survey_data["smoking"] == "yes":
            recs.append("Consider quitting smoking to reduce your heart disease risk.")
        if survey_data["age"] > 45:
            recs.append("Due to your age, regular heart check-ups and ECG are recommended.")
        if survey_data["bloodPressure"] > 140:
            recs.append("High blood pressure detected. Consider reducing salt intake and taking prescribed medication.")
    elif predictions["heartRisk"] == "Moderate":
        if survey_data["exercise"] == "never":
            recs.append("Starting moderate exercise like walking will help reduce heart risks.")
        if survey_data["cholesterol"] > 240:
            recs.append("Your cholesterol levels are slightly high. Avoid fried foods and increase fiber in your diet.")
    else:
        recs.append("Maintain a healthy lifestyle with regular exercise and a balanced diet to keep your heart healthy.")

    # Diabetes Risk Recommendations based on survey combinations
    if predictions["diabetesRisk"] == "High":
        if survey_data["sugarLevel"] > 130:
            recs.append("High sugar levels detected. Monitor your blood sugar regularly and consult a doctor.")
        if survey_data["weight"] > 85:
            recs.append("Your weight puts you at a higher risk. Consider a low-calorie, high-fiber diet and regular exercise.")
        if "diabetes" in survey_data["familyHistory"]:
            recs.append("Family history of diabetes. Regular blood sugar testing is recommended.")
    elif predictions["diabetesRisk"] == "Moderate":
        if survey_data["diet"] == "poor":
            recs.append("Improving your diet by reducing processed foods and sugar can help control diabetes risk.")
        if survey_data["exercise"] == "never":
            recs.append("A daily walk or light exercise can significantly reduce your diabetes risk.")
    else:
        recs.append("Your diabetes risk is low, but continue maintaining a healthy diet and exercise routine.")

    # Mental Health Risk Recommendations based on survey combinations
    if predictions["mentalRisk"] == "High":
        if survey_data["stress"] == "high":
            recs.append("High stress levels detected. Consider practicing mindfulness, yoga, or seeking professional therapy.")
        if survey_data["sleep"] == "<5":
            recs.append("Insufficient sleep can affect your mental health. Aim for 7-8 hours of sleep per night.")
    elif predictions["mentalRisk"] == "Moderate":
        if survey_data["sleep"] == "5-7":
            recs.append("Moderate sleep levels. Try to improve your rest quality to reduce stress.")
    else:
        recs.append("Your mental health risk is low. Keep up your current healthy lifestyle.")

    # Obesity Risk Recommendations based on survey combinations
    if predictions["obesityRisk"] == "High":
        if survey_data["activityLevel"] == "low":
            recs.append("A sedentary lifestyle can increase obesity risk. Try to increase your activity level with simple exercises.")
        if survey_data["diet"] == "poor":
            recs.append("Improving your diet by cutting back on processed foods and increasing fruits and vegetables can help.")
    elif predictions["obesityRisk"] == "Moderate":
        if survey_data["diet"] == "poor":
            recs.append("Consider consulting a nutritionist to create a healthier diet plan.")
        if survey_data["exercise"] == "never":
            recs.append("Begin light exercise to help reduce obesity risk, such as walking or swimming.")
    else:
        recs.append("Maintain a balanced diet and regular exercise routine to prevent obesity.")

    return recs

# Example usage in the survey analysis endpoint
@app.post("/api/survey")
async def analyze_risk(data: SurveyData):
    try:
        input_dict = data.dict()
        family = input_dict["familyHistory"]
        symptoms = input_dict["symptoms"]
        
        # Feature preparation for prediction
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

        # Model prediction
        raw_preds = model.predict(final_input)[0]
        
        # Decode predictions
        risks = {}
        for idx, key in enumerate(["heartRisk", "diabetesRisk", "mentalRisk", "obesityRisk"]):
            risks[key] = label_encoders[key].inverse_transform([raw_preds[idx]])[0]

        # Generate dynamic recommendations based on survey data and risk predictions
        recommendations = generate_dynamic_recommendations(risks, input_dict)
        
        result = {"data": input_dict, "result": {**risks, **recommendations}}

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
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(e)})


# User Registration
@app.post("/api/register")
async def register(request: RegisterRequest):
    user = users_collection.find_one({"username": request.username})
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = pwd_context.hash(request.password)
    new_user = {"username": request.username, "password": hashed_password}
    users_collection.insert_one(new_user)

    return {"message": "User registered successfully"}

# User Login
@app.post("/api/login")
async def login(request: LoginRequest):
    user = users_collection.find_one({"username": request.username})
    if not user or not pwd_context.verify(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


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
            "result": result["result"],
            "timestamp": datetime.utcnow()
        })

        return result

    except Exception as e:
        print("🔥 UPLOAD ERROR:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to process PDF", "detail": str(e)})

# GET /api/history — fetch user report history
@app.get("/api/history")
async def get_user_history(username: str = None):
    try:
        query_filter = {}
        if username:
            query_filter["username"] = username

        # Fetch user-specific health history records from MongoDB
        entries = list(collection.find(query_filter).sort("timestamp", -1))  # Filtered by username and sorted by timestamp

        for entry in entries:
            entry["_id"] = str(entry["_id"])  # Convert MongoDB ObjectId to string for frontend usage
        return entries  # Return the list of health report entries
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to fetch history", "detail": str(e)})
