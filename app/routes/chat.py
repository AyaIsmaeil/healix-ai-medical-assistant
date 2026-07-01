from fastapi import APIRouter
from pydantic import BaseModel

from app.services.marbert_service import predict_patient_symptoms

router = APIRouter(
    tags=["Patient Chat Analysis"]
)


class ChatMessageInput(BaseModel):
    text: str


@router.post("/analyze")
def analyze_patient_message(data: ChatMessageInput):
    symptoms = predict_patient_symptoms(data.text)

    return {
        "status": "success",
        "detected_symptoms": symptoms
    }
