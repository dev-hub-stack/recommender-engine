"""
Authentication Module for Master Group Recommendation System
Handles JWT token generation, validation, and password hashing
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-2024-mastergroup")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# HTTP Bearer token scheme
security = HTTPBearer()

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class User(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool = True

class UserInDB(User):
    password_hash: str

class LoginRequest(BaseModel):
    email: str
    password: str

# Database connection
def get_db_connection():
    """Get database connection"""
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:@localhost:5432/mastergroup_recommendations")
    return psycopg2.connect(DATABASE_URL)

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using bcrypt"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# User utilities
def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get user from database by email"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT id, email, password_hash, full_name, is_active FROM users WHERE email = %s",
            (email,)
        )
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user_data:
            return UserInDB(**user_data)
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with email and password"""
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

def update_last_login(email: str):
    """Update user's last login timestamp"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_login = %s WHERE email = %s",
            (datetime.now(), email)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error updating last login: {e}")

# JWT token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return TokenData(email=email)
    except JWTError:
        return None

# Dependency for protected routes
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    token_data = decode_access_token(token)
    
    if token_data is None or token_data.email is None:
        raise credentials_exception
    
    user = get_user_by_email(token_data.email)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return User(id=user.id, email=user.email, full_name=user.full_name, is_active=user.is_active)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
