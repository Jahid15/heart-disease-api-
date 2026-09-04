from typing import List

from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }
    )

    age: int = Field(..., ge=1, le=120, description="age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="chest pain type")
    trestbps: int = Field(..., ge=50, le=250, description="resting blood pressure in mm Hg")
    chol: int = Field(..., ge=100, le=700, description="serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="resting electrocardiographic result")
    thalach: int = Field(..., ge=60, le=250, description="maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="slope of the peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="number of major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="thalassemia result")


class PredictionResponse(BaseModel):
    heart_disease: bool
    probability: float
    label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    pipeline: List[str]
    features: List[str]
    classes: dict
    test_accuracy: float
    dataset: str
