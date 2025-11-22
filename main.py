
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Dự đoán Bỏ Học API")

model = joblib.load("dropout_model.pkl")

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

@app.post("/predict")
def predict(student: StudentInput):
    data = np.array([[student.TuitionFeesUpToDate, student.ScholarshipHolder, student.CurricularUnits1stSemGrade,
                      student.CurricularUnits1stSemApproved, student.CurricularUnits2ndSemGrade,
                      student.CurricularUnits2ndSemApproved, student.PreviousQualification,
                      student.MaritalStatus, student.AgeAtEnrollment, student.Gender, student.Course]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    risk = "Cao" if prob > 0.7 else "Trung bình" if prob > 0.4 else "Thấp"
    return {
        "Prediction": "Bỏ học" if pred == 1 else "Không bỏ học",
        "Dropout_Probability": round(float(prob), 4),
        "Risk_Level": risk
    }
