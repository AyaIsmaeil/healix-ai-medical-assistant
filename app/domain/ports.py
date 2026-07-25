"""
Healix - Domain Ports
الواجهات (Ports) التي تعتمد عليها طبقة التطبيق، وتُنفَّذ في الطبقات الخارجية.

استخدام ``Protocol`` يجعل الحقن (DI) بنيوياً: أي كائن يطابق الشكل يُقبل،
دون وراثة أو اقتران بالتنفيذ الملموس (تحقيق SOLID / قابلية الاختبار).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from app.domain.assessment import ClinicalFeatureSet, ValidationReport
from app.domain.confidence import ConfidenceAssessment
from app.domain.conversation import ConversationState
from app.domain.explanation import AssessmentExplanation
from app.domain.feature_encoder import EncodedFeatures
from app.domain.prediction import DiseasePredictionResult
from app.domain.specialty import SpecialtyRecommendation
from app.domain.urgency import UrgencyAssessment


@dataclass
class Completion:
    """نتيجة توليد نصّي من مزوّد الـ LLM."""

    text: str
    model: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """منفذ مزوّد نموذج اللغة."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        """توليد ردّ نصّي (يُفترض أنه JSON) من التعليمات المُعطاة."""
        ...


class SessionStore(Protocol):
    """منفذ تخزين جلسات المحادثة."""

    def get(self, session_id: str) -> Optional[ConversationState]:
        ...

    def get_or_create(self, session_id: Optional[str]) -> ConversationState:
        ...

    def save(self, state: ConversationState) -> None:
        ...


@runtime_checkable
class DiseasePredictorPort(Protocol):
    """منفذ التنبؤ بالمرض (Phase 3.4).

    يستهلك ``EncodedFeatures`` فقط — لا يقرأ ``ClinicalFeatureSet`` إطلاقاً،
    فلا يعرف شيئاً عن وكيل المقابلة أو FeatureValidator أو AssessmentFeatureBuilder.
    أي adapter (قاعدي الآن، XGBoost/RandomForest/CatBoost لاحقاً) يطابق هذا
    الشكل دون تغيير أي كود آخر بالنظام — استبدال الـadapter فقط.
    """

    def predict(self, encoded_features: EncodedFeatures) -> DiseasePredictionResult:
        """التنبؤ بالأمراض المحتملة من الميزات المُرمَّزة. يُعاد
        ``DiseasePredictionResult`` دائماً، حتى لو لم يُطابَق شيء."""
        ...


@runtime_checkable
class UrgencyClassifierPort(Protocol):
    """منفذ تقييم الاستعجال / الفرز الطبي (Triage) (Phase 3.5).

    يستهلك ``ClinicalFeatureSet`` مباشرة — لا ``EncodedFeatures`` — بخلاف
    ``DiseasePredictorPort``: قواعد الاستعجال تحتاج القيم الطبية الخام
    (الحرارة، الشدّة، احتمال الحمل...) لا تمثيلها المُرمَّز لنموذج تعلّم آلي.
    أي adapter (قاعدي الآن، نموذج ML لاحقاً) يطابق هذا الشكل دون تغيير أي
    كود آخر بالنظام — استبدال الـadapter فقط.
    """

    def classify(self, clinical_features: ClinicalFeatureSet) -> UrgencyAssessment:
        """تقييم درجة الاستعجال من الميزات السريرية المُتحقَّق منها. يُعاد
        ``UrgencyAssessment`` دائماً."""
        ...


@runtime_checkable
class SpecialtyRecommenderPort(Protocol):
    """منفذ توصية التخصّص الطبي (Phase 3.6).

    يستهلك ``ClinicalFeatureSet`` مباشرة — لا ``EncodedFeatures`` — بنفس شكل
    ``UrgencyClassifierPort``. يستقبل أيضاً ``prediction_result`` اختيارياً:
    قرار معماري موثَّق (انظر ``rule_based_specialty_recommender.py``) — قواعد
    التوصية تعطي الأولوية لأعلى مرض مُتنبَّأ به من DiseasePredictor عند
    توفّره، وهذا يتطلّب مدخلاً لا يوفّره ClinicalFeatureSet وحده. توسيع أدنى
    ضروري، لا يغيّر الطابع الأساسي "ClinicalFeatureSet لا EncodedFeatures".
    أي adapter (قاعدي الآن، نموذج ML لاحقاً) يطابق هذا الشكل دون تغيير أي
    كود آخر بالنظام — استبدال الـadapter فقط.
    """

    def recommend(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: Optional[DiseasePredictionResult] = None,
    ) -> SpecialtyRecommendation:
        """توصية التخصّص الطبي الأنسب. يُعاد ``SpecialtyRecommendation`` دائماً."""
        ...


@runtime_checkable
class ConfidenceEstimatorPort(Protocol):
    """منفذ تقدير موثوقية التقييم الكامل (Phase 3.7).

    آخر مرحلة بخطّ التقييم — تقرأ كل ما أُنتج قبلها وتُصدر حكماً على الثقة
    فقط: لا تتنبّأ بمرض، لا تغيّر الاستعجال، لا تغيّر التخصّص. تستقبل مخرجات
    المراحل السابقة صراحةً (لا ``EncodedFeatures``) — القيم السريرية وتقرير
    التحقّق والتنبؤ والاستعجال والتخصّص — فالثقة حكم كلّي على المنظومة، لا
    على تمثيل مُرمَّز لنموذج واحد.

    ``validation`` يُمرَّر صراحةً رغم كونه متاحاً بـ``clinical_features.validation``:
    توضيح للعقد وتيسير للاختبار (نفس روح تمرير كل منفذ ما يحتاجه بدقّة).

    أي adapter (قاعدي الآن، مُعايِر ثقة ML لاحقاً) يطابق هذا الشكل دون تغيير
    أي كود آخر بالنظام — استبدال الـadapter فقط.
    """

    def estimate(
        self,
        clinical_features: ClinicalFeatureSet,
        validation: ValidationReport,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
    ) -> ConfidenceAssessment:
        """تقدير موثوقية التقييم الكامل. يُعاد ``ConfidenceAssessment`` دائماً."""
        ...


@runtime_checkable
class AssessmentExplainerPort(Protocol):
    """منفذ التفسير النهائي للتقييم (Phase 3.8) — المرحلة الأخيرة بالخطّ.

    يشرح ما حُسِب سلفاً بالعربية فقط: لا يُشخّص، ولا يُعدّل أي تنبؤ/استعجال/
    تخصّص/ثقة. يستقبل مخرجات المراحل السابقة صراحةً (لا ``EncodedFeatures``،
    ولا رسائل مريض خام) — التفسير يعتمد على المعلومات المنظَّمة المحسوبة.

    ``confidence`` جزء من المدخل لأنّ ``requires_human_review`` يوجّه التوصية
    (يجب أن تُشجّع صراحةً على تقييم مهني عند طلب المراجعة البشرية).

    التنفيذ الحالي مبنيّ على LLM لكنّه يضمن الإرجاع دائماً (تدهور حتمي لطيف
    عند فشل الـLLM/التحليل). أي adapter آخر يطابق هذا الشكل دون تغيير أي كود
    آخر بالنظام — استبدال الـadapter فقط.
    """

    def explain(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
        urgency: UrgencyAssessment,
        specialty: SpecialtyRecommendation,
        confidence: ConfidenceAssessment,
    ) -> AssessmentExplanation:
        """تفسير التقييم الكامل بالعربية. يُعاد ``AssessmentExplanation`` دائماً."""
        ...
