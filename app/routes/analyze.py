"""
Healix - Symptom Extraction Route
نقطة الـ API لاستخراج الأعراض من نص عربي باستخدام نموذج MARBERT (NER).

المدخل:  { "text": "..." }
المخرج:  { "symptoms": [ { "text": "...", "negated": false, "confidence": 0.97 } ] }
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_symptom_extractor
from app.exceptions import InferenceError, ModelNotLoadedError
from app.schemas.symptom import ExtractRequest, ExtractResponse, SymptomOut
from app.services.symptom_extractor import SymptomExtractor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["استخراج الأعراض"])


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="استخراج الأعراض من نص",
    description="يستخرج الأعراض (الإيجابية والمنفية) من النص باستخدام نموذج MARBERT المُدرّب.",
)
async def extract_symptoms(
    request: ExtractRequest,
    extractor: SymptomExtractor = Depends(get_symptom_extractor),
) -> ExtractResponse:
    logger.info("طلب استخراج الأعراض: %s...", request.text[:50])
    try:
        symptoms = extractor.extract(request.text)
        return ExtractResponse(
            symptoms=[
                SymptomOut(text=s.text, negated=s.negated, confidence=s.confidence)
                for s in symptoms
            ]
        )
    except ModelNotLoadedError as exc:
        logger.error("النموذج غير مُحمّل: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except InferenceError as exc:
        logger.error("فشل الاستدلال: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
