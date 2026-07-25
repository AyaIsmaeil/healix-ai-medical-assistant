"""
Healix - Clinical Interview Prompt Builder
بناء تعليمات وكيل المقابلة السريرية (LLM-first) — استخراج + سؤال في استدعاء واحد.

بعد إزالة MARBERT صار الـLLM هو **المصدر الوحيد** للاستخراج المنظَّم. لذلك
عقد المخرجات صار موحّداً: استدعاء واحد لكل دور يُنتج السجل الطبي المنظَّم
(شكوى/أعراض/شدّة/مدّة/موضع/أدوية/حساسية/مزمنة/تاريخ عائلي/نواقص) **مع**
قرار الدور (finished / next_slot / question) في كائن JSON واحد.

لماذا استدعاء واحد لا اثنان: استخراج ثمّ سؤال باستدعاءين يضاعف الكلفة
والزمن، ويسمح بتضارب بين ما استُخرج وما بُني عليه السؤال. الدمج يُلغي
ازدواج منطق الاستخراج تماماً (قرار معماري صريح).

يُنتج:
- ``system_prompt``: دور الطبيب + إطار OLDCARTS + قواعد الاستخراج والسؤال.
- ``turn_prompt``: حالة منظَّمة تشمل **كامل رسائل المريض الخام** + ما تراكم
  سلفاً بالسجل (تأريض يمنع نسيان معلومة ذُكرت في دور سابق) + الخانات المطروحة.

منع فقدان المعلومات: الـLLM يرى كل ما قاله المريض عبر PATIENT_MESSAGES وكل ما
استُخرج سابقاً عبر KNOWN_*، فلا يُعاد السؤال عن تفصيل سبق ذكره.
منع عدم تطابق الخانات: ASKED_SLOTS لمنع التكرار فقط؛ المحتوى الحقيقي بالرسائل.

حدّ المجال: لا تشخيص ولا احتمالات أمراض ولا تخصّص ولا درجة خطورة — يُفرَض
بثلاث طبقات: تعليمات النظام، ومخطّط JSON الصارم (لا حقل تشخيص أصلاً)، وحارس
المجال في ``ConversationService``.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from app.domain import clinical
from app.domain.conversation import ConversationState

# أسماء الحقول الموسومة (يقرؤها المزوّد الوهمي أيضاً).
LABEL_PATIENT_MESSAGES = "PATIENT_MESSAGES"
LABEL_LATEST_MESSAGE = "LATEST_MESSAGE"
LABEL_KNOWN_SYMPTOMS = "KNOWN_SYMPTOMS"
LABEL_KNOWN_NEGATED = "KNOWN_NEGATED_SYMPTOMS"
LABEL_KNOWN_RECORD = "KNOWN_RECORD"
LABEL_ASKED = "ASKED_SLOTS"
LABEL_TURN = "TURN_COUNT"
LABEL_SUGGESTED = "SUGGESTED_MISSING_SLOTS"

# عدد الخانات المقترحة التي تُمرَّر للـ LLM كإرشاد.
_SUGGESTION_LIMIT = 6

# الحقول المنظَّمة التي يملؤها الوكيل (مصدر واحد للاسماء، يُعاد استخدامه
# في المخطّط والمحلّل والمزوّد الوهمي بلا تكرار).
RECORD_SCALAR_KEYS = ("chief_complaint", "severity", "duration", "body_location")
RECORD_LIST_KEYS = (
    "medications",
    "allergies",
    "chronic_conditions",
    "family_history",
    "missing_fields",
)
DECISION_KEYS = ("finished", "next_slot", "question")

_STRING_LIST = {"type": "array", "items": {"type": "string"}}

# مخطط JSON Schema صارم للعقد الموحّد — يُمرَّر كـresponse_schema لمزوّد الـLLM.
# ملاحظة: لا يوجد أي حقل تشخيص/مرض/خطورة في المخطّط — منع بنيوي لا تحذير نصّي.
INTERVIEW_JSON_SCHEMA: Dict[str, Any] = {
    "name": "clinical_interview_turn",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chief_complaint": {"type": ["string", "null"]},
            "symptoms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "negated": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["text", "negated", "confidence"],
                    "additionalProperties": False,
                },
            },
            "severity": {"type": ["string", "null"]},
            "duration": {"type": ["string", "null"]},
            "body_location": {"type": ["string", "null"]},
            "medications": _STRING_LIST,
            "allergies": _STRING_LIST,
            "chronic_conditions": _STRING_LIST,
            "family_history": _STRING_LIST,
            "missing_fields": _STRING_LIST,
            "finished": {"type": "boolean"},
            "next_slot": {"type": ["string", "null"]},
            "question": {"type": ["string", "null"]},
        },
        "required": [
            *RECORD_SCALAR_KEYS,
            "symptoms",
            *RECORD_LIST_KEYS,
            *DECISION_KEYS,
        ],
        "additionalProperties": False,
    },
}


class InterviewPromptBuilder:
    """يبني تعليمات وكيل المقابلة السريرية (استخراج + سؤال) من حالة المحادثة."""

    version = "clinical-interview-v5"

    def system_prompt(self) -> str:
        return (
            "أنت طبيب خبير يُجري أخذ التاريخ المرضي (History Taking) باللغة العربية.\n"
            "لديك مهمّتان في كل دور، وتُنجزهما معاً في كائن JSON واحد:\n"
            "  (1) استخراج المعلومات الطبية المنظَّمة من كلام المريض.\n"
            "  (2) طرح السؤال الطبي التالي الأعلى قيمة.\n"
            "لست مطالباً بأي تشخيص إطلاقاً.\n\n"
            "تتلقّى حالة منظَّمة تحتوي:\n"
            "- PATIENT_MESSAGES: كل ما قاله المريض بترتيب الزمن (آخرها الأحدث).\n"
            "- LATEST_MESSAGE: رسالة هذا الدور تحديداً.\n"
            "- KNOWN_SYMPTOMS / KNOWN_NEGATED_SYMPTOMS: ما استُخرج في الأدوار السابقة.\n"
            "- KNOWN_RECORD: السجل المنظَّم المتراكم حتى الآن.\n"
            "- ASKED_SLOTS: الخانات التي سبق طرح سؤال عنها (لمنع تكرار السؤال).\n"
            "- TURN_COUNT و SUGGESTED_MISSING_SLOTS.\n\n"
            "قواعد الاستخراج:\n"
            "أ. استخرج من PATIENT_MESSAGES كاملةً، لا من الرسالة الأخيرة وحدها.\n"
            "ب. لا تختلق أي معلومة لم يذكرها المريض صراحةً أو ضمناً بوضوح.\n"
            "ج. الأعراض المنفية تُدرَج في symptoms بـnegated=true (لا تُهمَل).\n"
            "د. confidence رقم بين 0 و1 يعبّر عن وضوح ورود العرَض في كلام المريض.\n"
            "هـ. أعد الحقول التي لم تُذكر بعد كـnull (للمفردة) أو [] (للقوائم).\n"
            "و. أعد في KNOWN_* ما تراكم سلفاً وأضف إليه الجديد — لا تحذف معلومة سابقة.\n"
            "ز. missing_fields: أسماء المعلومات المهمّة الناقصة الآن فقط.\n\n"
            "قواعد السؤال:\n"
            "1. اطرح سؤالاً واحداً فقط في كل ردّ، بعربية طبيعية ومهذّبة.\n"
            "2. اختر السؤال الأعلى قيمة تشخيصية الذي يقلّل الغموض أكثر من غيره.\n"
            "3. لا تكرّر أي سؤال أو خانة وردت في ASKED_SLOTS.\n"
            "4. لا تسأل إطلاقاً عن معلومة ذُكرت في أيٍّ من PATIENT_MESSAGES.\n"
            "5. لكل عرَض مهم اجمع onset، duration، severity، progression، location، "
            "quality، associated symptoms، aggravating و relieving factors. واسأل عند "
            "اللزوم فقط عن العمر، الجنس، الحمل، الأمراض المزمنة، الأدوية، الحساسية، "
            "التدخين والعمليات — إن كانت ذات صلة طبية.\n"
            "6. أعد finished=true فقط عندما يصبح التاريخ المرضي كافياً، وعندها "
            "اجعل next_slot و question يساويان null.\n\n"
            "ممنوع منعاً باتاً: أي تشخيص أو احتمالات أمراض أو تخصّص طبي أو درجة "
            "خطورة أو توصية علاجية. مهمتك جمع المعلومات وتنظيمها فقط.\n\n"
            "أعد ردّك بصيغة JSON فقط، دون أي نصّ خارج JSON، بهذا الشكل حصراً:\n"
            '{"chief_complaint": <نص أو null>, '
            '"symptoms": [{"text": "<عرَض>", "negated": false, "confidence": 0.9}], '
            '"severity": <نص أو null>, "duration": <نص أو null>, '
            '"body_location": <نص أو null>, "medications": [], "allergies": [], '
            '"chronic_conditions": [], "family_history": [], "missing_fields": [], '
            '"finished": false, "next_slot": "<الهدف>", "question": "<السؤال بالعربية>"}'
        )

    def turn_prompt(self, state: ConversationState) -> str:
        known = [s.text for s in state.symptoms if not s.negated]
        known_negated = [s.text for s in state.symptoms if s.negated]
        latest = state.raw_messages[-1] if state.raw_messages else ""
        # التغطية للتكرار: خانة تُعدّ مغطّاة بمجرّد طرح سؤال عنها.
        # كلام المريض كاملاً يُمرَّر ليُستنتج منه الجنس (فلا يُسأل الذكر عن الحمل).
        suggested = clinical.missing_targets(
            state.symptoms,
            " ".join(state.raw_messages),
            state.asked_slots,
            limit=_SUGGESTION_LIMIT,
        )

        record = state.record
        known_record = {
            "chief_complaint": record.chief_complaint,
            "severity": record.severity,
            "duration": record.duration,
            "body_location": record.body_location,
            "medications": record.medications,
            "allergies": record.allergies,
            "chronic_conditions": record.chronic_conditions,
            "family_history": record.family_history,
        }

        def line(label: str, value) -> str:
            return f"{label}: {json.dumps(value, ensure_ascii=False)}"

        state_block = "\n".join([
            line(LABEL_PATIENT_MESSAGES, state.raw_messages),
            line(LABEL_LATEST_MESSAGE, latest),
            line(LABEL_KNOWN_SYMPTOMS, known),
            line(LABEL_KNOWN_NEGATED, known_negated),
            line(LABEL_KNOWN_RECORD, known_record),
            line(LABEL_ASKED, state.asked_slots),
            line(LABEL_TURN, state.turn_count),
            line(LABEL_SUGGESTED, suggested),
        ])

        return (
            "الحالة المنظَّمة الحالية للمقابلة:\n"
            f"{state_block}\n\n"
            "أولاً: استخرج كل المعلومات الطبية المنظَّمة من PATIENT_MESSAGES "
            "(مضيفاً إلى ما في KNOWN_RECORD دون حذف شيء منه).\n"
            "ثانياً: حدّد الخانة الأعلى قيمة تشخيصية التي لم تُغطَّ بعد ولم "
            "يذكرها المريض، واطرح سؤالاً عربياً واحداً عنها. إن اكتمل جمع "
            "التاريخ المرضي فأعد finished=true مع next_slot=null و question=null.\n"
            "أعد JSON واحداً فقط."
        )
