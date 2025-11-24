from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model ngay khi khởi động (nhanh hơn load mỗi request)
model_path = os.path.join(os.path.dirname(__file__), "dropout_model.pkl")
model = joblib.load(model_path)

app = FastAPI(title="Dự đoán Sinh Viên Bỏ Học", version="1.0")


class StudentInput(BaseModel):
    TuitionFeesUpToDate: int
    ScholarshipHolder: int
    CurricularUnits1stSemGrade: float
    CurricularUnits1stSemApproved: int
    CurricularUnits2ndSemGrade: float
    CurricularUnits2ndSemApproved: int
    PreviousQualification: int
    MaritalStatus: int
    AgeAtEnrollment: int
    Gender: int
    Course: int


@app.get("/")
def home():
    return {"message": "API dự đoán bỏ học đang chạy! Gửi POST đến /predict"}


@app.post("/predict")
def predict(student: StudentInput):
    # Chuyển input thành numpy array
    data = np.array([[
        student.TuitionFeesUpToDate,
        student.ScholarshipHolder,
        student.CurricularUnits1stSemGrade,
        student.CurricularUnits1stSemApproved,
        student.CurricularUnits2ndSemGrade,
        student.CurricularUnits2ndSemApproved,
        student.PreviousQualification,
        student.MaritalStatus,
        student.AgeAtEnrollment,
        student.Gender,
        student.Course
    ]])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]  # Xác suất lớp 1 (Dropout)

    return {
        "Prediction": "Bỏ học" if pred == 1 else "Không bỏ học",
        "Dropout_Probability": round(float(prob), 4),
        "Risk_Level": "Cao" if prob > 0.7 else "Trung bình" if prob > 0.4 else "Thấp"
    }
