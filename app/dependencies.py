"""
Healix - Dependencies
مزوّدو الاعتمادية (Dependency Injection) لطبقة الـ API.

تُبنى الكائنات مرّة واحدة عند بدء التشغيل وتُخزَّن في ``app.state``،
ثمّ تُحقَن هنا عبر ``Depends`` دون أي متغيّرات عامة على مستوى الوحدة.
"""

from fastapi import Request

from app.domain.feature_encoder import FeatureEncoder
from app.domain.feature_validator import FeatureValidator
from app.domain.symptom_evidence_encoder import SymptomEvidenceEncoder
from app.services.composite_evidence_concept_extractor import CompositeEvidenceConceptExtractor
from app.domain.ports import (
    AssessmentExplainerPort,
    ConfidenceEstimatorPort,
    DiseasePredictorPort,
    SpecialtyRecommenderPort,
    UrgencyClassifierPort,
)
from app.exceptions import (
    AssessmentError,
    ConversationError,
    FeatureValidationError,
    ModelNotLoadedError,
)
from app.services.assessment_feature_builder import AssessmentFeatureBuilder
from app.services.conversation_service import ConversationService
from app.services.speech_to_text import SpeechToTextService


def get_speech_to_text_service(request: Request) -> SpeechToTextService:
    """إرجاع خدمة التفريغ النصّي (Whisper) المُهيّأة عند بدء التشغيل."""
    service = getattr(request.app.state, "speech_service", None)
    if service is None:
        raise ModelNotLoadedError("خدمة تفريغ الصوت (Whisper) غير مُهيّأة.")
    return service


def get_conversation_service(request: Request) -> ConversationService:
    """إرجاع محرك المحادثة المُهيّأ عند بدء التشغيل."""
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise ConversationError("محرك المحادثة غير مُهيّأ.")
    return service


def get_assessment_feature_builder(request: Request) -> AssessmentFeatureBuilder:
    """إرجاع منسّق بناء ميزات التقييم المُهيّأ عند بدء التشغيل."""
    builder = getattr(request.app.state, "assessment_feature_builder", None)
    if builder is None:
        raise AssessmentError("محرك التقييم غير مُهيّأ.")
    return builder


def get_feature_validator(request: Request) -> FeatureValidator:
    """إرجاع طبقة التحقّق من الميزات المُهيّأة عند بدء التشغيل."""
    validator = getattr(request.app.state, "feature_validator", None)
    if validator is None:
        raise FeatureValidationError("طبقة التحقّق من الميزات غير مُهيّأة.")
    return validator


def get_feature_encoder(request: Request) -> FeatureEncoder:
    """إرجاع مشفِّر الميزات المُهيّأ عند بدء التشغيل."""
    encoder = getattr(request.app.state, "feature_encoder", None)
    if encoder is None:
        raise AssessmentError("مشفِّر الميزات غير مُهيّأ.")
    return encoder


def get_symptom_evidence_encoder(request: Request) -> SymptomEvidenceEncoder:
    """إرجاع مُرمِّز ربط الأعراض بأدلة DDXPlus المُهيّأ عند بدء التشغيل."""
    encoder = getattr(request.app.state, "symptom_evidence_encoder", None)
    if encoder is None:
        raise AssessmentError("مُرمِّز ربط الأعراض بالأدلة غير مُهيّأ.")
    return encoder


def get_evidence_concept_extractor(request: Request) -> CompositeEvidenceConceptExtractor:
    """إرجاع مُستخلِص مفاهيم الأدلة الهجين (LLM + قواعد) المُهيّأ عند بدء التشغيل."""
    extractor = getattr(request.app.state, "evidence_concept_extractor", None)
    if extractor is None:
        raise AssessmentError("مُستخلِص مفاهيم الأدلة غير مُهيّأ.")
    return extractor


def get_disease_predictor(request: Request) -> DiseasePredictorPort:
    """إرجاع مُتنبِّئ المرض المُهيّأ عند بدء التشغيل (Adapter الحالي: قاعدي —
    استبداله لاحقاً بـXGBoost/RandomForest/CatBoost لا يغيّر هذه الدالة)."""
    predictor = getattr(request.app.state, "disease_predictor", None)
    if predictor is None:
        raise AssessmentError("مُتنبِّئ المرض غير مُهيّأ.")
    return predictor


def get_urgency_classifier(request: Request) -> UrgencyClassifierPort:
    """إرجاع مُصنِّف الاستعجال المُهيّأ عند بدء التشغيل (Adapter الحالي:
    قاعدي — استبداله لاحقاً بنموذج ML لا يغيّر هذه الدالة)."""
    classifier = getattr(request.app.state, "urgency_classifier", None)
    if classifier is None:
        raise AssessmentError("مُصنِّف الاستعجال غير مُهيّأ.")
    return classifier


def get_specialty_recommender(request: Request) -> SpecialtyRecommenderPort:
    """إرجاع مُوصي التخصّص الطبي المُهيّأ عند بدء التشغيل (Adapter الحالي:
    قاعدي — استبداله لاحقاً بنموذج ML لا يغيّر هذه الدالة)."""
    recommender = getattr(request.app.state, "specialty_recommender", None)
    if recommender is None:
        raise AssessmentError("مُوصي التخصّص غير مُهيّأ.")
    return recommender


def get_confidence_estimator(request: Request) -> ConfidenceEstimatorPort:
    """إرجاع مُقدِّر موثوقية التقييم المُهيّأ عند بدء التشغيل (Adapter الحالي:
    قاعدي — استبداله لاحقاً بمُعايِر ثقة ML لا يغيّر هذه الدالة)."""
    estimator = getattr(request.app.state, "confidence_estimator", None)
    if estimator is None:
        raise AssessmentError("مُقدِّر الموثوقية غير مُهيّأ.")
    return estimator


def get_assessment_explainer(request: Request) -> AssessmentExplainerPort:
    """إرجاع مُفسِّر التقييم المُهيّأ عند بدء التشغيل (يستخدم مزوّد LLM مستقلاً
    بعقد JSON خاص — استبداله لاحقاً بأي adapter تفسير لا يغيّر هذه الدالة)."""
    explainer = getattr(request.app.state, "assessment_explainer", None)
    if explainer is None:
        raise AssessmentError("مُفسِّر التقييم غير مُهيّأ.")
    return explainer
