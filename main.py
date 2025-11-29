from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load mô hình từ file .pkl
model = joblib.load("dropout_model.pkl")  # Thay path nếu cần

app = FastAPI(title="Student Dropout Prediction API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def get_home():
    return FileResponse("static/index.html")

# Định nghĩa schema input dựa trên features từ dataset của bạn
class PredictionInput(BaseModel):
    TuitionFeesUpToDate: int  # 1 or 0
    ScholarshipHolder: int  # 1 or 0
    CurricularUnits1stSemGrade: float
    CurricularUnits1stSemApproved: int
    CurricularUnits2ndSemGrade: float
    CurricularUnits2ndSemApproved: int
    PreviousQualification: int
    MaritalStatus: int
    AgeAtEnrollment: int
    Gender: int  # 1 or 0
    Course: int

# Endpoint để dự đoán (POST request)
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Chuyển input thành DataFrame để predict
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])  # Tạo DF từ dict
        
        # Dự đoán (0: Not Dropout, 1: Dropout)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Xác suất Dropout
        
        result = "Dropout" if prediction == 1 else "Not Dropout"
        return {
            "prediction": result,
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Endpoint test (GET)
@app.get("/")
def root():
    return {"message": "API is running. Use /predict for predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

