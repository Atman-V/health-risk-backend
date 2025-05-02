from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from . import database, models

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Settings
SECRET_KEY = "your-secret-key"  # Store securely in environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Helper function to create a JWT token
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Login and registration schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

# Router
router = APIRouter()

# Route for user registration
@router.post("/register")
async def register(request: RegisterRequest):
    db = database.get_db()  # Get database session

    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.username == request.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Hash the password
    hashed_password = pwd_context.hash(request.password)

    # Create new user and save to DB
    new_user = models.User(username=request.username, password=hashed_password)
    db.add(new_user)
    db.commit()

    return {"message": "User registered successfully"}

# Route for user login
@router.post("/login")
async def login(request: LoginRequest):
    db = database.get_db()  # Get database session

    # Fetch the user from the database
    user = db.query(models.User).filter(models.User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify password
    if not pwd_context.verify(request.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {"access_token": access_token, "token_type": "bearer"}

