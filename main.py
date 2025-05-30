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
    familyHistory: list[str]
    symptoms: list[str]
    bloodPressure: float
    sugarLevel: float
    cholesterol: float
    mentalHealth: str
    activityLevel: str

# Generate Recommendations for health risks
def generate_recommendations(predictions):
    full_recs = {
        "heart": {
            "High": ["Recommendation: Consider quitting smoking to reduce heart disease risk.", 
                     "Recommendation: Regular heart check-ups are recommended due to high risk.",
                     "Recommendation: Engage in heart-healthy exercise, such as walking or swimming.",
                     "Recommendation: Consider medications to help manage heart health, as advised by a healthcare provider."],
            "Moderate": ["Recommendation: Start moderate exercise like walking to reduce heart risks.", 
                         "Recommendation: Monitor your cholesterol and reduce fat intake.",
                         "Recommendation: Focus on a diet rich in fruits, vegetables, and whole grains."],
            "Low": ["Recommendation: Keep up with a healthy lifestyle and regular exercise."]
        },
        "diabetes": {
            "High": ["Recommendation: Monitor your blood sugar levels regularly.",
                     "Recommendation: Consider a low-carb, high-fiber diet to manage diabetes.",
                     "Recommendation: Speak with a healthcare provider about managing insulin levels."],
            "Moderate": ["Recommendation: Improve your diet by reducing sugar intake and increasing fiber.",
                         "Recommendation: Engage in regular physical activity to help manage your blood sugar."],
            "Low": ["Recommendation: Keep maintaining a healthy lifestyle to prevent diabetes."]
        },
        "mental": {
            "High": ["Recommendation: Consider seeing a therapist to address mental health concerns.",
                     "Recommendation: Practice stress-reduction techniques like mindfulness or meditation.",
                     "Recommendation: Consider joining support groups for additional emotional support."],
            "Moderate": ["Recommendation: Try to get more sleep and reduce stress levels to improve mental health.",
                         "Recommendation: Regular physical activity can help reduce stress and improve mood."],
            "Low": ["Recommendation: Continue managing your mental health with regular sleep and stress management."]
        },
        "obesity": {
            "High": ["Recommendation: Begin a fitness regimen to improve overall health and reduce obesity risk.",
                     "Recommendation: Follow a balanced diet with less processed food and more vegetables.",
                     "Recommendation: Consider consulting a nutritionist or weight management specialist."],
            "Moderate": ["Recommendation: Start incorporating exercise into your routine to avoid further weight gain.",
                         "Recommendation: Focus on portion control and balanced meals."],
            "Low": ["Recommendation: Maintain a healthy weight by continuing your current healthy lifestyle."]
        }
    }
    
    result = {}
    for key in ["heart", "diabetes", "mental", "obesity"]:
        level = predictions[f"{key}Risk"]
        result[f"{key}Advice"] = full_recs[key].get(level, [])
    return result

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

# POST /api/survey — form submission (for risk analysis)
@app.post("/api/survey")
async def analyze_risk(data: SurveyData):
    try:
        input_dict = data.dict()
        
        # familyHistory and symptoms are already arrays, so no need to split them
        family = input_dict["familyHistory"]
        symptoms = input_dict["symptoms"]

        # Prepare input data for prediction
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
        
        # Transform family history and symptoms
        fam_vec = mlb_family.transform([family])
        sym_vec = mlb_symptoms.transform([symptoms])
        
        final_input = np.hstack([input_arr, fam_vec[0], sym_vec[0]]).reshape(1, -1)

        # Model prediction
        raw_preds = model.predict(final_input)[0]
        
        risks = {}
        for idx, key in enumerate(["heartRisk", "diabetesRisk", "mentalRisk", "obesityRisk"]):
            risks[key] = label_encoders[key].inverse_transform([raw_preds[idx]])[0]

        # Generate recommendations
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
            "result": result["result"],
            "timestamp": datetime.utcnow()
        })

        return result

    except Exception as e:
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
