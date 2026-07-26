from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os
import re
import time
import sqlite3
import uuid
from datetime import datetime, timedelta

# =========================
# Config
# =========================

IMG_SIZE = 224
MODEL_ACCURACY = 85.5
CONFIDENCE_THRESHOLD = 60.0

MODEL_PATH = "model/plant_leaf_model_finetuned.keras"
CLASS_NAMES_PATH = "model/class_names.json"

DB_PATH = "prediction_logs.db"
UPLOAD_DIR = "uploads"

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_KEY_FOR_LEAFVISION_AI")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Plant Leaf Identification API",
    description="API สำหรับจำแนกชนิดพืชจากภาพใบไม้ด้วย MobileNetV2 Fine-tuned Model",
    version="4.1.0",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip()

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# =========================
# Database
# =========================

def get_connection():
    return sqlite3.connect(DB_PATH)


def add_column_if_not_exists(
    cursor,
    table_name: str,
    column_name: str,
    column_definition: str,
):
    try:
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )
    except sqlite3.OperationalError:
        pass


def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content_type TEXT,
            file_size INTEGER,
            image_url TEXT,
            status TEXT,
            prediction TEXT,
            display_name TEXT,
            confidence REAL,
            top3_json TEXT,
            model_accuracy REAL,
            confidence_threshold REAL,
            processing_time_seconds REAL,
            created_at TEXT,
            user_id INTEGER,
            username TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            is_active INTEGER DEFAULT 1,
            created_at TEXT,
            last_login TEXT
        )
        """
    )

    add_column_if_not_exists(cursor, "prediction_logs", "image_url", "TEXT")
    add_column_if_not_exists(cursor, "prediction_logs", "user_id", "INTEGER")
    add_column_if_not_exists(cursor, "prediction_logs", "username", "TEXT")

    add_column_if_not_exists(cursor, "prediction_logs", "verified_status", "TEXT DEFAULT 'pending'")
    add_column_if_not_exists(cursor, "prediction_logs", "actual_class", "TEXT")
    add_column_if_not_exists(cursor, "prediction_logs", "is_correct", "INTEGER")
    add_column_if_not_exists(cursor, "prediction_logs", "verification_note", "TEXT")
    add_column_if_not_exists(cursor, "prediction_logs", "verified_by", "INTEGER")
    add_column_if_not_exists(cursor, "prediction_logs", "verified_by_username", "TEXT")
    add_column_if_not_exists(cursor, "prediction_logs", "verified_at", "TEXT")

    add_column_if_not_exists(cursor, "users", "email", "TEXT")
    add_column_if_not_exists(cursor, "users", "is_active", "INTEGER DEFAULT 1")
    add_column_if_not_exists(cursor, "users", "last_login", "TEXT")

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
        ON users(username)
        """
    )

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email
        ON users(email)
        WHERE email IS NOT NULL
        """
    )

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]

    if user_count == 0:
        default_password = "admin12345"
        password_hash = pwd_context.hash(default_password)

        cursor.execute(
            """
            INSERT INTO users (
                username,
                email,
                password_hash,
                role,
                is_active,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "admin",
                "admin@leafvision.local",
                password_hash,
                "admin",
                1,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

        print("✅ Default admin created")
        print("username: admin")
        print("email: admin@leafvision.local")
        print("password: admin12345")
    else:
        cursor.execute(
            """
            UPDATE users
            SET email = COALESCE(email, ?),
                is_active = COALESCE(is_active, 1)
            WHERE username = ?
            """,
            ("admin@leafvision.local", "admin"),
        )

    conn.commit()
    conn.close()


def insert_prediction_log(
    filename: str,
    content_type: str,
    file_size: int,
    image_url: str,
    status_value: str,
    prediction: str,
    display_name: str,
    confidence: float,
    top3: list,
    processing_time_seconds: float,
    user_id: int,
    username: str,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_logs (
            filename,
            content_type,
            file_size,
            image_url,
            status,
            prediction,
            display_name,
            confidence,
            top3_json,
            model_accuracy,
            confidence_threshold,
            processing_time_seconds,
            created_at,
            user_id,
            username
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            content_type,
            file_size,
            image_url,
            status_value,
            prediction,
            display_name,
            confidence,
            json.dumps(top3, ensure_ascii=False),
            MODEL_ACCURACY,
            CONFIDENCE_THRESHOLD,
            processing_time_seconds,
            datetime.now().isoformat(timespec="seconds"),
            user_id,
            username,
        ),
    )

    conn.commit()
    conn.close()


init_database()

# =========================
# Auth helpers
# =========================

def is_valid_email(email: str):
    if not email:
        return False

    email = email.strip().lower()
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    return re.match(pattern, email) is not None


def normalize_email(email: str):
    return email.strip().lower()


def verify_password(plain_password: str, password_hash: str):
    return pwd_context.verify(plain_password, password_hash)


def get_password_hash(password: str):
    return pwd_context.hash(password)


def get_user_by_username(username: str):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, email, password_hash, role, is_active, created_at, last_login
        FROM users
        WHERE username = ?
        """,
        (username,),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def get_user_by_email(email: str):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, email, password_hash, role, is_active, created_at, last_login
        FROM users
        WHERE email = ?
        """,
        (normalize_email(email),),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def get_user_by_username_or_email(identifier: str):
    identifier = identifier.strip()
    email = normalize_email(identifier)

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, email, password_hash, role, is_active, created_at, last_login
        FROM users
        WHERE username = ? OR email = ?
        """,
        (identifier, email),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def authenticate_user(identifier: str, password: str):
    user = get_user_by_username_or_email(identifier)

    if not user:
        return None

    if int(user.get("is_active", 1)) != 1:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET last_login = ?
        WHERE id = ?
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            user["id"],
        ),
    )

    conn.commit()
    conn.close()

    return user


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="ไม่สามารถยืนยันตัวตนได้ กรุณา login ใหม่",
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)

    if user is None:
        raise credentials_exception

    if int(user.get("is_active", 1)) != 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="บัญชีนี้ถูกปิดใช้งาน",
        )

    return {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email"),
        "role": user["role"],
    }


def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ต้องเป็น Admin เท่านั้นจึงจะเข้าถึงส่วนนี้ได้",
        )

    return current_user

def require_verifier_or_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") not in ["admin", "verifier"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ต้องเป็น Verifier หรือ Admin เท่านั้นจึงจะเข้าถึงส่วนนี้ได้",
        )

    return current_user

# =========================
# Load Model and Class Names
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ class names ที่: {CLASS_NAMES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in range(len(class_names))]

print("✅ Model loaded successfully")
print(f"✅ Model path: {MODEL_PATH}")
print(f"✅ Number of classes: {len(class_names)}")
print("✅ Classes:", class_names)
print(f"✅ SQLite log database: {DB_PATH}")

# =========================
# Helper Functions
# =========================

def validate_image_file(file: UploadFile):
    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="รองรับเฉพาะไฟล์ภาพ .jpg, .jpeg, .png, .webp เท่านั้น",
        )


def save_uploaded_image(filename: str, image_bytes: bytes):
    safe_ext = os.path.splitext(filename or "uploaded.jpg")[1].lower()

    if safe_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        safe_ext = ".jpg"

    unique_filename = f"{uuid.uuid4().hex}{safe_ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(save_path, "wb") as f:
        f.write(image_bytes)

    image_url = f"/uploads/{unique_filename}"

    return image_url


def preprocess_image(image_bytes: bytes):
    """
    สำคัญ:
    ไม่ใช้ mobilenet_v2.preprocess_input ที่ backend
    เพราะ preprocess_input ถูกฝังอยู่ในโมเดลตอน train แล้ว
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="ไม่สามารถอ่านไฟล์ภาพได้ กรุณาอัปโหลดภาพใหม่",
        )

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def get_top_predictions(predictions, top_k=3):
    top_indices = predictions.argsort()[-top_k:][::-1]

    top_predictions = []

    for index in top_indices:
        top_predictions.append(
            {
                "name": class_names[index],
                "confidence": round(float(predictions[index]) * 100, 2),
            }
        )

    return top_predictions


def predict_image_from_bytes(
    file: UploadFile,
    image_bytes: bytes,
    current_user: dict,
):
    start_time = time.time()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ภาพว่างเปล่า กรุณาอัปโหลดใหม่",
        )

    image_array = preprocess_image(image_bytes)

    try:
        predictions = model.predict(image_array, verbose=0)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดระหว่างทำนายผล: {str(e)}",
        )

    top3 = get_top_predictions(predictions, top_k=3)
    best_prediction = top3[0]
    processing_time = round(time.time() - start_time, 3)

    if best_prediction["confidence"] < CONFIDENCE_THRESHOLD:
        response = {
            "status": "low_confidence",
            "prediction": "Unknown",
            "display_name": "ไม่สามารถระบุชนิดพืชได้อย่างมั่นใจ",
            "confidence": best_prediction["confidence"],
            "message": "โมเดลยังไม่มั่นใจพอ กรุณาอัปโหลดภาพใบไม้ที่ชัดเจนขึ้น ถ่ายเฉพาะใบ และลดฉากหลังรบกวน",
            "top3": top3,
            "model_accuracy": MODEL_ACCURACY,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "processing_time_seconds": processing_time,
        }
    else:
        response = {
            "status": "success",
            "prediction": best_prediction["name"],
            "display_name": best_prediction["name"],
            "confidence": best_prediction["confidence"],
            "message": "ทำนายสำเร็จ",
            "top3": top3,
            "model_accuracy": MODEL_ACCURACY,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "processing_time_seconds": processing_time,
        }

    image_url = save_uploaded_image(file.filename or "uploaded.jpg", image_bytes)

    insert_prediction_log(
        filename=file.filename or "unknown",
        content_type=file.content_type or "unknown",
        file_size=len(image_bytes),
        image_url=image_url,
        status_value=response["status"],
        prediction=response["prediction"],
        display_name=response["display_name"],
        confidence=response["confidence"],
        top3=response["top3"],
        processing_time_seconds=response["processing_time_seconds"],
        user_id=current_user["id"],
        username=current_user["username"],
    )

    return response

# =========================
# Routes
# =========================

@app.get("/")
def home():
    return {
        "message": "Plant Leaf AI Backend is running",
        "version": "4.1.0",
        "features": [
            "register",
            "login with username or email",
            "admin user management",
            "role management",
            "active status management",
            "single image prediction",
            "multi image prediction support from frontend queue",
            "prediction logs with uploaded images",
            "JWT authentication",
            "low confidence handling",
        ],
        "model": "plant_leaf_model_finetuned.keras",
        "model_accuracy": MODEL_ACCURACY,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "number_of_classes": len(class_names),
        "classes": class_names,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "class_names_loaded": True,
        "number_of_classes": len(class_names),
        "database": DB_PATH,
        "uploads": UPLOAD_DIR,
    }


@app.post("/auth/register")
def register(payload: dict):
    username = str(payload.get("username", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if not username:
        raise HTTPException(
            status_code=400,
            detail="กรุณากรอก username",
        )

    if len(username) < 3:
        raise HTTPException(
            status_code=400,
            detail="username ต้องมีอย่างน้อย 3 ตัวอักษร",
        )

    if not is_valid_email(email):
        raise HTTPException(
            status_code=400,
            detail="รูปแบบ email ไม่ถูกต้อง เช่น example@gmail.com",
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="password ต้องมีอย่างน้อย 8 ตัวอักษร",
        )

    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="password ยาวเกินไป กรุณาใช้ไม่เกิน 72 bytes",
        )

    existing_username = get_user_by_username(username)

    if existing_username:
        raise HTTPException(
            status_code=409,
            detail="username นี้ถูกใช้งานแล้ว",
        )

    existing_email = get_user_by_email(email)

    if existing_email:
        raise HTTPException(
            status_code=409,
            detail="email นี้ถูกใช้งานแล้ว",
        )

    password_hash = get_password_hash(password)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO users (
            username,
            email,
            password_hash,
            role,
            is_active,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            email,
            password_hash,
            "user",
            1,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    conn.commit()
    conn.close()

    return {
        "message": "register สำเร็จ กรุณาเข้าสู่ระบบ",
        "user": {
            "username": username,
            "email": email,
            "role": "user",
        },
    }


@app.post("/auth/login")
def login(payload: dict):
    identifier = str(
        payload.get("identifier") or payload.get("username") or payload.get("email") or ""
    ).strip()

    password = str(payload.get("password", ""))

    if not identifier or not password:
        raise HTTPException(
            status_code=400,
            detail="กรุณากรอก username/email และ password",
        )

    user = authenticate_user(identifier, password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="username/email หรือ password ไม่ถูกต้อง",
        )

    access_token = create_access_token(
        data={
            "sub": user["username"],
            "role": user["role"],
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user.get("email"),
            "role": user["role"],
        },
    }


@app.get("/auth/me")
def auth_me(current_user: dict = Depends(get_current_user)):
    return {
        "authenticated": True,
        "user": current_user,
    }


@app.get("/admin/users")
def get_admin_users(current_user: dict = Depends(require_admin)):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            username,
            email,
            role,
            is_active,
            created_at,
            last_login
        FROM users
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    users = [dict(row) for row in rows]

    return {
        "total": len(users),
        "users": users,
    }


@app.patch("/admin/users/{user_id}/role")
def update_user_role(
    user_id: int,
    payload: dict,
    current_user: dict = Depends(require_admin),
):
    new_role = str(payload.get("role", "")).strip().lower()

    allowed_roles = ["user", "verifier", "admin"]

    if new_role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail="role ต้องเป็น user, verifier หรือ admin เท่านั้น",
        )

    if user_id == current_user["id"] and new_role != "admin":
        raise HTTPException(
            status_code=400,
            detail="ไม่สามารถลด role ของบัญชีตัวเองจาก admin ได้",
        )

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )

    user = cursor.fetchone()

    if user is None:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="ไม่พบ user นี้",
        )

    cursor.execute(
        """
        UPDATE users
        SET role = ?
        WHERE id = ?
        """,
        (new_role, user_id),
    )

    conn.commit()

    cursor.execute(
        """
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )

    updated_user = dict(cursor.fetchone())
    conn.close()

    return {
        "message": "เปลี่ยน role สำเร็จ",
        "user": updated_user,
    }


@app.patch("/admin/users/{user_id}/active")
def update_user_active_status(
    user_id: int,
    payload: dict,
    current_user: dict = Depends(require_admin),
):
    is_active = payload.get("is_active")

    if is_active not in [0, 1, True, False]:
        raise HTTPException(
            status_code=400,
            detail="is_active ต้องเป็น 0 หรือ 1",
        )

    is_active_value = 1 if bool(is_active) else 0

    if user_id == current_user["id"] and is_active_value == 0:
        raise HTTPException(
            status_code=400,
            detail="ไม่สามารถปิดใช้งานบัญชีตัวเองได้",
        )

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )

    user = cursor.fetchone()

    if user is None:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="ไม่พบ user นี้",
        )

    cursor.execute(
        """
        UPDATE users
        SET is_active = ?
        WHERE id = ?
        """,
        (is_active_value, user_id),
    )

    conn.commit()

    cursor.execute(
        """
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )

    updated_user = dict(cursor.fetchone())
    conn.close()

    return {
        "message": "เปลี่ยน active status สำเร็จ",
        "user": updated_user,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    validate_image_file(file)
    image_bytes = await file.read()

    return predict_image_from_bytes(file, image_bytes, current_user)


@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="กรุณาอัปโหลดไฟล์อย่างน้อย 1 ไฟล์",
        )

    results = []

    for file in files:
        try:
            validate_image_file(file)
            image_bytes = await file.read()
            result = predict_image_from_bytes(file, image_bytes, current_user)

            results.append(
                {
                    "filename": file.filename,
                    "ok": True,
                    "result": result,
                }
            )

        except Exception as e:
            results.append(
                {
                    "filename": file.filename,
                    "ok": False,
                    "error": str(e),
                }
            )

    return {
        "total_files": len(files),
        "results": results,
    }


@app.get("/logs")
def get_logs(
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    logs = []

    for row in rows:
        item = dict(row)

        try:
            item["top3"] = json.loads(item.pop("top3_json"))
        except Exception:
            item["top3"] = []

        logs.append(item)

    return {
        "total": len(logs),
        "logs": logs,
    }


@app.get("/logs/summary")
def get_logs_summary(current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prediction_logs")
    total_predictions = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT status, COUNT(*)
        FROM prediction_logs
        GROUP BY status
        """
    )
    status_counts = dict(cursor.fetchall())

    cursor.execute(
        """
        SELECT prediction, COUNT(*)
        FROM prediction_logs
        GROUP BY prediction
        ORDER BY COUNT(*) DESC
        """
    )
    prediction_counts = dict(cursor.fetchall())

    cursor.execute("SELECT AVG(confidence) FROM prediction_logs")
    avg_confidence = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT username, COUNT(*)
        FROM prediction_logs
        GROUP BY username
        ORDER BY COUNT(*) DESC
        """
    )
    user_counts = dict(cursor.fetchall())

    conn.close()

    return {
        "total_predictions": total_predictions,
        "status_counts": status_counts,
        "prediction_counts": prediction_counts,
        "user_counts": user_counts,
        "average_confidence": round(avg_confidence or 0, 2),
    }

@app.get("/verifier/logs")
def get_verifier_logs(
    limit: int = 50,
    verified_status: str = "pending",
    current_user: dict = Depends(require_verifier_or_admin),
):
    allowed_status = ["pending", "approved", "rejected", "all"]

    if verified_status not in allowed_status:
        raise HTTPException(
            status_code=400,
            detail="verified_status ต้องเป็น pending, approved, rejected หรือ all",
        )

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if verified_status == "all":
        cursor.execute(
            """
            SELECT *
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
    else:
        cursor.execute(
            """
            SELECT *
            FROM prediction_logs
            WHERE COALESCE(verified_status, 'pending') = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (verified_status, limit),
        )

    rows = cursor.fetchall()
    conn.close()

    logs = []

    for row in rows:
        item = dict(row)

        try:
            item["top3"] = json.loads(item.pop("top3_json"))
        except Exception:
            item["top3"] = []

        logs.append(item)

    return {
        "total": len(logs),
        "logs": logs,
    }

@app.patch("/verifier/logs/{log_id}/review")
def review_prediction_log(
    log_id: int,
    payload: dict,
    current_user: dict = Depends(require_verifier_or_admin),
):
    verified_status = str(payload.get("verified_status", "")).strip().lower()
    actual_class = str(payload.get("actual_class", "")).strip()
    verification_note = str(payload.get("verification_note", "")).strip()

    if verified_status not in ["approved", "rejected"]:
        raise HTTPException(
            status_code=400,
            detail="verified_status ต้องเป็น approved หรือ rejected",
        )

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM prediction_logs
        WHERE id = ?
        """,
        (log_id,),
    )

    log = cursor.fetchone()

    if log is None:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="ไม่พบ prediction log นี้",
        )

    log_dict = dict(log)
    model_prediction = log_dict.get("prediction")

    if verified_status == "approved":
        final_actual_class = actual_class or model_prediction
        is_correct = 1
    else:
        if not actual_class:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="กรุณาระบุ actual_class เมื่อ Reject",
            )

        final_actual_class = actual_class
        is_correct = 0

    cursor.execute(
        """
        UPDATE prediction_logs
        SET verified_status = ?,
            actual_class = ?,
            is_correct = ?,
            verification_note = ?,
            verified_by = ?,
            verified_by_username = ?,
            verified_at = ?
        WHERE id = ?
        """,
        (
            verified_status,
            final_actual_class,
            is_correct,
            verification_note,
            current_user["id"],
            current_user["username"],
            datetime.now().isoformat(timespec="seconds"),
            log_id,
        ),
    )

    conn.commit()

    cursor.execute(
        """
        SELECT *
        FROM prediction_logs
        WHERE id = ?
        """,
        (log_id,),
    )

    updated_log = dict(cursor.fetchone())

    try:
        updated_log["top3"] = json.loads(updated_log.pop("top3_json"))
    except Exception:
        updated_log["top3"] = []

    conn.close()

    return {
        "message": "บันทึกผล verification สำเร็จ",
        "log": updated_log,
    }