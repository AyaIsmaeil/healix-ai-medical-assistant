"""
Healix - Clinical Interview Schemas
نماذج الطلب والاستجابة لمحرك المقابلة السريرية (طبقة الـ API).

توافق خلفي مقصود: حقول الاستجابة القديمة (``finished`` و``question``
و``next_slot`` و``symptoms``...) باقية كما هي بنفس أسمائها ودلالاتها، وأُضيفت
إليها حقول السجل الطبي المنظَّم الجديدة. الإضافة فقط — لا إعادة تسمية ولا حذف —
حتى لا ينكسر أي مستهلك حالي (تكامل Laravel).

``interview_complete`` و``next_question`` مرآتان صريحتان لـ``finished``
و``question``: العقد الجديد يستخدمهما، والقديم يبقى صالحاً. تُملأ الاثنتان دائماً
بنفس القيمة، فلا مصدر حقيقة مزدوج.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SymptomOut(BaseModel):
    text: str
    negated: bool
    # أُضيف مع معمارية LLM-first (إضافة غير كاسرة؛ المستهلك القديم يتجاهله).
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # إثبات المصدر: هل ذكره المريض فعلاً أم استنتجه النموذج؟
    source: Optional[str] = Field(
        default=None,
        description="patient_explicit | llm_inferred — مصدر هذا العرَض",
    )
    evidence: Optional[str] = Field(
        default=None, description="نصّ المريض الحرفي المُسنِد للعرَض",
    )


class RedFlagOut(BaseModel):
    """علم أحمر أُطلق — يُعرَض للتفسير والتدقيق، لا للتشخيص."""

    rule_id: str
    name_ar: str
    name_en: str
    risk_level: str
    evidence: Optional[str] = None


class InterviewTurnRequest(BaseModel):
    """رسالة مريض في مقابلة أخذ التاريخ المرضي."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="رسالة المريض بالعربية",
        examples=["أعاني من صداع وحرارة"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="معرّف الجلسة (يُترك فارغاً في أول رسالة ويُعاد استخدامه بعدها)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"text": "أعاني من صداع وحرارة", "session_id": None}
        }
    }


class InterviewTurnResponse(BaseModel):
    """استجابة الدور: السجل الطبي المنظَّم + سؤال عربي واحد أو إشارة الانتهاء."""

    # --- العقد الأصلي (يبقى كما هو، لا يتغيّر) ---
    session_id: str
    finished: bool
    next_slot: Optional[str] = None
    question: Optional[str] = None
    turn: int
    status: str
    symptoms: List[SymptomOut] = Field(default_factory=list)

    # --- السجل الطبي المنظَّم (إضافة معمارية LLM-first) ---
    chief_complaint: Optional[str] = Field(
        default=None, description="الشكوى الرئيسية كما استخلصها وكيل المقابلة"
    )
    severity: Optional[str] = Field(default=None, description="شدّة العرَض كما ذُكرت")
    duration: Optional[str] = Field(default=None, description="مدّة العرَض كما ذُكرت")
    body_location: Optional[str] = Field(
        default=None, description="موضع العرَض في الجسم كما ذُكر"
    )
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(
        default_factory=list, description="المعلومات المهمّة الناقصة الآن"
    )

    # --- مرآتان للعقد الجديد (نفس قيمة finished/question دائماً) ---
    interview_complete: bool = Field(
        default=False, description="مرآة لـfinished بتسمية العقد الجديد"
    )
    next_question: Optional[str] = Field(
        default=None, description="مرآة لـquestion بتسمية العقد الجديد"
    )

    # --- طبقة السلامة (إضافة غير كاسرة) ---
    # ملاحظة عقد: هذه الحقول **لا تُشخّص**. تشير إلى احتمال استعجال وتوصي
    # برعاية مهنية/طارئة فقط — ولا تذكر مرضاً ولا احتماله.
    emergency_detected: bool = Field(
        default=False,
        description="هل أُطلق علم أحمر يستوجب رعاية عاجلة؟",
    )
    risk_level: str = Field(
        default="none", description="none | low | urgent | immediate",
    )
    red_flags: List[RedFlagOut] = Field(
        default_factory=list, description="الأعلام الحمراء التي أُطلقت",
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="التوجيه الموصى به بالعربية عند وجود طارئ (لا تشخيص)",
    )
    safety_screening_degraded: bool = Field(
        default=False,
        description="تعذّر الاستخراج (فشل الـLLM) وصدر الحكم من الفحص الحتمي وحده",
    )
    primary_complaint: Optional[str] = Field(
        default=None,
        description=(
            "الشكوى الرئيسية الحالية كما قرّرها محرّك الأولوية السريرية. "
            "تختلف عن chief_complaint: تلك نصّ استخرجه النموذج، وهذه قرار "
            "المحرّك عن محور الأسئلة الآن، ويُعاد حسابه بعد كل رسالة."
        ),
    )
    session_restarted: bool = Field(
        default=False,
        description=(
            "معرّف الجلسة المُرسَل غير معروف أو انتهت صلاحيته، فبدأ سجلّ جديد. "
            "على العميل إعلام المريض بدل متابعة الحوار وكأنّ السياق محفوظ."
        ),
    )
