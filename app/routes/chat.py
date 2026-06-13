from fastapi import APIRouter
from pydantic import BaseModel

from app.service.nlp_service import extract_symptoms
from app.service.diagnosis_service import diagnose

router = APIRouter()


class ChatRequest(BaseModel):
    user_input: str


@router.post("/")
def chat(request: ChatRequest):

    symptoms = extract_symptoms(request.user_input)
    result = diagnose(symptoms)

    return {
        "input": request.user_input,
        "symptoms": symptoms,
        "diagnosis": result
    }