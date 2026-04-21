from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Placement Prediction API", description="Predict student placement and expected salary")

clf = joblib.load('artifacts/xgb_classifier.pkl')
reg = joblib.load('artifacts/linear_regression.pkl')

class StudentData(BaseModel):
    gender: str
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    entrance_exam_score: int
    technical_skill_score: int
    soft_skill_score: int
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: float
    backlogs: int
    extracurricular_activities: str

@app.get("/")
def root():
    return {"message": "Placement Prediction API is running"}

@app.post("/predict")
def predict(student: StudentData):
    try:
        input_df = pd.DataFrame([student.dict()])
        placement = int(clf.predict(input_df)[0])
        if placement == 1:
            salary = float(reg.predict(input_df)[0])
        else:
            salary = 0.0
        return {
            "placement_status": placement,
            "predicted_salary_lpa": salary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# python -m fastapi dev api.py