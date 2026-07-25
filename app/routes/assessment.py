"""
Healix - Assessment Route
نقطة الـAPI لمحرك التقييم:
Phase 3.1 بناء الميزات، 3.2 التحقّق، 3.3 الترميز، 3.4 التنبؤ القاعدي بالمرض،
3.5 تقييم الاستعجال القاعدي (Triage)، 3.6 توصية التخصّص القاعدية، 3.7 تقدير
الموثوقية القاعدي، 3.8 التفسير النهائي بالعربية (الخطّ الكامل).

المدخل: لقطة كاملة من محادثة منتهية (رسائل خام + الأعراض الجاهزة من المقابلة).
الخدمة مستقلة (stateless) — لا تُقرأ من جلسة محرّك المقابلة الداخلية، بل
تستقبل كل ما يلزم بالطلب نفسه (نفس مبدأ بقية نقاط الـAPI الحالية).

التسلسل: AssessmentFeatureBuilder.build() → FeatureValidator.validate() →
FeatureEncoder.encode() → DiseasePredictorPort.predict() →
UrgencyClassifierPort.classify() → SpecialtyRecommenderPort.recommend() →
ConfidenceEstimatorPort.estimate() → AssessmentExplainerPort.explain() — ثماني
استدعاءات (لا تعديل على منطق أيٍّ منها). كل من ``urgency_classifier`` و
``specialty_recommender`` يعمل على ``features`` (ClinicalFeatureSet المُتحقَّق
منه) مباشرة، لا على ``encoded``. ``specialty_recommender`` يستقبل أيضاً
``prediction_result`` (قرار معماري موثَّق بـ``SpecialtyRecommenderPort``).
``confidence_estimator`` يقرأ كل ما أُنتج قبله ويُصدر حكماً على الثقة فقط.
``explainer`` هو المرحلة الأخيرة: يشرح ما حُسِب سلفاً بالعربية فقط (لا تشخيص،
لا تعديل تنبؤ/استعجال/تخصّص/ثقة) — يستدعي LLM لكنه يضمن الإرجاع دائماً
(تدهور حتمي لطيف عند فشل الـLLM/التحليل، فلا يُسقط التقييم المحسوب سلفاً).
كل الـPorts مُحقَنة عبر المنفذ لا الصنف الملموس — استبدال أي Adapter لا يغيّر
هذا الملف إطلاقاً.

المخرج: ClinicalFeatureSet فعلي مُتحقَّق منه ومُرمَّز + تنبؤات المرض القاعدية
+ تقييم الاستعجال القاعدي + توصية التخصّص القاعدية + تقدير الموثوقية القاعدي
+ التفسير النهائي بالعربية (لا بيانات وهمية) — الخطّ الكامل.

ملاحظة: FeatureValidationError (ترث من HealixError مباشرة، لا من
AssessmentError) لا تُلتقط بـexcept هنا عمداً — تتسرّب للمعالج العام
الموجود أصلاً بـ main.py (@app.exception_handler(HealixError))، تحقيقاً
لـ"إعادة استخدام معالجة الأخطاء الحالية، بلا معالج جديد".
"""

import asyncio
import logging
from dataclasses import asdict
from functools import partial

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import (
    get_assessment_explainer,
    get_assessment_feature_builder,
    get_confidence_estimator,
    get_disease_predictor,
    get_feature_encoder,
    get_feature_validator,
    get_specialty_recommender,
    get_urgency_classifier,
)
from app.domain.clinical_record import ClinicalRecord
from app.domain.conversation import Symptom
from app.domain.feature_encoder import FeatureEncoder
from app.domain.feature_validator import FeatureValidator
from app.domain.ports import (
    AssessmentExplainerPort,
    ConfidenceEstimatorPort,
    DiseasePredictorPort,
    SpecialtyRecommenderPort,
    UrgencyClassifierPort,
)
from app.exceptions import AssessmentError, FeatureExtractionError
from app.schemas.assessment import (
    AssessmentExplanationOut,
    AssessmentRequest,
    AssessmentResponse,
    ClinicalFeatureSetOut,
    ConfidenceAssessmentOut,
    DiseasePredictionResultOut,
    SpecialtyRecommendationOut,
    UrgencyAssessmentOut,
)
from app.services.assessment_feature_builder import AssessmentFeatureBuilder

logger = logging.getLogger(__name__)
router = APIRouter(tags=["محرك التقييم"])


@router.post(
    "/assessment/run",
    response_model=AssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="بناء ميزات التقييم السريري من محادثة منتهية",
    description=(
        "يستقبل لقطة كاملة من محادثة منتهية (رسائل خام + أعراض المقابلة "
        "الجاهزة)، يبني ClinicalFeatureSet، يتحقّق منه، يرمّزه، يتنبّأ "
        "بالأمراض المحتملة، يقيّم درجة الاستعجال، يوصي بالتخصّص، يقدّر "
        "موثوقية التقييم، ويُنتج تفسيراً عربياً نهائياً (Adapters قاعدية + LLM)."
    ),
)
async def run_assessment(
    request: AssessmentRequest,
    builder: AssessmentFeatureBuilder = Depends(get_assessment_feature_builder),
    validator: FeatureValidator = Depends(get_feature_validator),
    encoder: FeatureEncoder = Depends(get_feature_encoder),
    predictor: DiseasePredictorPort = Depends(get_disease_predictor),
    urgency_classifier: UrgencyClassifierPort = Depends(get_urgency_classifier),
    specialty_recommender: SpecialtyRecommenderPort = Depends(get_specialty_recommender),
    confidence_estimator: ConfidenceEstimatorPort = Depends(get_confidence_estimator),
    explainer: AssessmentExplainerPort = Depends(get_assessment_explainer),
) -> AssessmentResponse:
    logger.info(
        "طلب تقييم | session=%s | %d رسالة | %d عرَض",
        request.session_id, len(request.raw_messages), len(request.symptoms),
    )

    symptoms = [
        Symptom(text=s.text, negated=s.negated, confidence=s.confidence)
        for s in request.symptoms
    ]

    # السجل المنظَّم من المقابلة (اختياري): إرساله يمنع إعادة استخراج ما ورد
    # فيه بالـLLM ويملأ التاريخ المرضي. غيابه يُبقي السلوك القديم كما هو.
    interview_record = (
        ClinicalRecord(**request.interview_record.model_dump())
        if request.interview_record is not None
        else None
    )

    # استخراج قاعدي سريع لكنه قد يستدعي الـLLM (شبكة) — نُشغّله خارج حلقة الأحداث.
    loop = asyncio.get_running_loop()
    try:
        features = await loop.run_in_executor(
            None,
            partial(
                builder.build, request.session_id, request.raw_messages, symptoms,
                interview_record,
            ),
        )
        features = await loop.run_in_executor(None, partial(validator.validate, features))
        encoded = await loop.run_in_executor(None, partial(encoder.encode, features))
        prediction_result = await loop.run_in_executor(None, partial(predictor.predict, encoded))
        urgency = await loop.run_in_executor(
            None, partial(urgency_classifier.classify, features)
        )
        specialty = await loop.run_in_executor(
            None, partial(specialty_recommender.recommend, features, prediction_result)
        )
        confidence = await loop.run_in_executor(
            None,
            partial(
                confidence_estimator.estimate,
                features,
                features.validation,
                prediction_result,
                urgency,
                specialty,
            ),
        )
        # المرحلة الأخيرة: تفسير عربي لما حُسِب سلفاً فقط. تستدعي LLM (شبكة)
        # لكنها تضمن الإرجاع دائماً (تدهور حتمي لطيف) — لا تُسقط التقييم.
        explanation = await loop.run_in_executor(
            None,
            partial(
                explainer.explain,
                features,
                prediction_result,
                urgency,
                specialty,
                confidence,
            ),
        )
    except FeatureExtractionError as exc:
        logger.error("فشل استخراج الميزات: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc
    except AssessmentError as exc:
        logger.error("خطأ بمحرك التقييم: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return AssessmentResponse(
        features=ClinicalFeatureSetOut.model_validate(asdict(features)),
        predictions=DiseasePredictionResultOut.model_validate(asdict(prediction_result)),
        # تحويل صريح لـ.value (لا asdict/model_validate) — نفس أسلوب تسلسل
        # UrgencyLevel(str, Enum) الآمن، بدل الاعتماد الضمني على وراثة str.
        urgency=UrgencyAssessmentOut(
            level=urgency.level.value,
            score=urgency.score,
            explanation=urgency.explanation,
        ),
        specialty=SpecialtyRecommendationOut(
            specialty=specialty.specialty,
            confidence=specialty.confidence,
            explanation=specialty.explanation,
        ),
        confidence=ConfidenceAssessmentOut(
            overall_confidence=confidence.overall_confidence,
            requires_human_review=confidence.requires_human_review,
            explanation=confidence.explanation,
        ),
        explanation=AssessmentExplanationOut(
            summary=explanation.summary,
            medical_reasoning=explanation.medical_reasoning,
            recommendation=explanation.recommendation,
            disclaimer=explanation.disclaimer,
        ),
    )
