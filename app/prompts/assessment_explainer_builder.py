"""
Healix - Assessment Explainer Prompt Builder (Phase 3.8)
بناء تعليمات الـLLM لتفسير التقييم المُحسَب سلفاً بالعربية.

بنفس أسلوب ``InterviewPromptBuilder`` و``AssessmentExtractionPromptBuilder``:
نظام + تعليمات دور، تسميات JSON واضحة، عقد إخراج ثابت الشكل (أربعة مفاتيح
نصّية دائماً)، ومخطّط JSON Schema صارم يُمرَّر لمزوّد الـLLM.

⚠️ مبدأ حاسم: الـLLM يتلقّى **معلومات منظَّمة فقط** (شكوى رئيسية، تنبؤات،
استعجال، تخصّص، ثقة، ميزات مُتحقَّقة) — لا رسائل مريض خام إطلاقاً. الخدمة
(AssessmentExplainer) هي من يبني هذا القاموس المنظَّم وتحرص ألا يتضمّن أي
نصّ محادثة خام. هذا الباني مجرّد مُنسِّق: يحوّل القاموس الجاهز إلى نصّ موسوم.

الـLLM لا يُشخّص، ولا يخترع أمراضاً غير الموجودة بالتنبؤات، ولا يعدّل
استعجالاً أو تخصّصاً أو ثقة — يشرح فقط.
"""

from __future__ import annotations

import json
from typing import Any, Dict

# مخطط JSON Schema صارم يطابق مفاتيح التفسير الأربعة — يُمرَّر لمزوّد الـLLM
# (response_schema بـQwenOpenRouterProvider) بدل عقد المقابلة الافتراضي.
# بنفس شكل _INTERVIEW_SCHEMA / EXTRACTION_JSON_SCHEMA (name/strict/schema).
EXPLANATION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "assessment_explanation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "medical_reasoning": {"type": "string"},
            "recommendation": {"type": "string"},
            "disclaimer": {"type": "string"},
        },
        "required": ["summary", "medical_reasoning", "recommendation", "disclaimer"],
        "additionalProperties": False,
    },
}


class AssessmentExplainerPromptBuilder:
    """يبني تعليمات تفسير التقييم من قاموس منظَّم جاهز (لا رسائل خام)."""

    version = "assessment-explain-v1"

    def system_prompt(self) -> str:
        return (
            "أنت مساعد طبي يشرح نتيجة تقييم مبدئي للمريض بلغة عربية واضحة "
            "ومطمئنة. أنت لا تُشخّص، ولا تُجري أي تقييم جديد — تشرح فقط ما "
            "حُسِب سلفاً وتلقّيته منظَّماً.\n\n"
            "القواعد الإلزامية:\n"
            "1. لا تخترع أي مرض. اذكر فقط الأمراض الواردة في DISEASE_PREDICTIONS "
            "إن وُجدت؛ وإن كانت فارغة فلا تذكر أي مرض.\n"
            "2. لا تُعدّل مستوى الاستعجال (URGENCY) — انقله كما هو.\n"
            "3. لا تُعدّل التخصّص (SPECIALTY) — انقله كما هو.\n"
            "4. لا تُعدّل درجة الثقة (CONFIDENCE) — لا تعطِ رقماً مختلفاً.\n"
            "5. إذا كان requires_human_review يساوي true، يجب أن تُشجّع التوصية "
            "صراحةً على مراجعة طبيب مختصّ لتقييم مهني.\n"
            "6. أمثلة على التوصية: «التوجّه إلى قسم الطوارئ فوراً»، «حجز موعد مع "
            "التخصّص المقترح»، «متابعة الأعراض في المنزل».\n"
            "7. يجب أن ينصّ disclaimer دائماً على أنّ هذا التقييم ليس تشخيصاً "
            "طبياً نهائياً.\n\n"
            "أعد ردّك بصيغة JSON فقط، دون أي نصّ خارجه، وبالمفاتيح الأربعة "
            "التالية حصراً (كلّها نصوص عربية غير فارغة):\n"
            '{"summary": "...", "medical_reasoning": "...", '
            '"recommendation": "...", "disclaimer": "..."}'
        )

    def explanation_prompt(self, structured: Dict[str, Any]) -> str:
        """يحوّل القاموس المنظَّم الجاهز إلى نصّ موسوم. لا يقرأ ولا يمرّر أي
        رسائل مريض خام — مسؤولية بناء ``structured`` بلا نصّ خام تقع على
        الخدمة المستدعية."""

        def line(label: str, value: Any) -> str:
            return f"{label}: {json.dumps(value, ensure_ascii=False)}"

        block = "\n".join(line(label, value) for label, value in structured.items())
        return (
            "معلومات التقييم المنظَّمة (محسوبة سلفاً — لا تُعدّلها):\n"
            f"{block}\n\n"
            "اشرح هذا التقييم للمريض بالعربية وفق القواعد، وأعد JSON فقط "
            "بالمفاتيح الأربعة."
        )
