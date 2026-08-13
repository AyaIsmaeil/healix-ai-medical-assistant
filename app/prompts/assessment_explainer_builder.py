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

    version = "assessment-explain-v2-hybrid"

    def system_prompt(self, include_rag: bool = False) -> str:
        base = (
            "أنت مساعد طبي يشرح نتيجة تقييم مبدئي للمريض بلغة عربية واضحة "
            "ومطمئنة. أنت لا تُشخّص، ولا تُجري أي تقييم جديد — تشرح فقط ما "
            "حُسِب سلفاً وتلقّيته منظَّماً.\n\n"
            "القواعد الإلزامية:\n"
            "1. لا تخترع أي مرض. اذكر فقط الأمراض الواردة في DISEASE_PREDICTIONS "
            "إن وُجدت؛ وإن كانت فارغة فلا تذكر أي مرض.\n"
            "2. لا تُعدّل مستوى الاستعجال (URGENCY) — انقله كما هو.\n"
            "3. لا تُعدّل التخصّص (SPECIALTY) — انقله كما هو.\n"
            "4. لا تُعدّل درجة الثقة (CONFIDENCE) — لا تعطِ رقماً مختلفاً.\n"
            "5. **summary و recommendation و disclaimer مُولَّدة حتمياً — "
            "أنت مسؤول عن medical_reasoning فقط.** لا تذكر العمر أو الجنس أو "
            "الشدّة إلا إن وردت في IMPORTANT_VALIDATED_FEATURES — ولا تخترع "
            "قيماً مختلفة.\n"
            "6. إذا كان requires_human_review يساوي true، اذكر في medical_reasoning "
            "ضرورة مراجعة طبيب مختصّ.\n\n"
            "أسلوب medical_reasoning تحديداً (هذا الحقل الوحيد الذي تكتبه فعلياً "
            "ويُقرأ لاحقاً — الباقي يُستبدَل بنصّ حتمي بصرف النظر عمّا تكتبه):\n"
            "- جملتان إلى ثلاث جمل قصيرة وواضحة، لا فقرة واحدة مكدَّسة بكل التفاصيل.\n"
            "- بصيغة طبيب يشرح استنتاجه لمريض، لا تقرير آلي يسرد احتمالات ونسباً "
            "بجفاف. اربط الاحتمالات بالأعراض المذكورة، لا تُعدّدها فقط.\n"
            "- ممنوع أي عبارة تكشف أنّ الكاتب نموذج لغوي: 'بصفتي مساعد ذكاء "
            "اصطناعي'، 'كنموذج لغوي لا أستطيع...'، 'يسعدني مساعدتك'، أو تكرار "
            "تنويه 'لست طبيباً' — التنويه الطبي يصدر مرّة واحدة بحقل disclaimer "
            "المنفصل، تكراره هنا حشو آلي واضح.\n"
            "- مثال جيّد: 'الأعراض التي ذكرتها — الحمّى مع السعال المستمرّ — "
            "تتوافق مع التهاب تنفّسي أكثر من غيره وفق البيانات المتاحة. هذا "
            "احتمال مبدئي فقط، والفحص السريري هو ما يحسم التشخيص.'\n"
            "- مثال سيّئ (تجنّبه): 'بناءً على البيانات المُدخَلة، تشير النتائج "
            "الإحصائية إلى أنّ الاحتمالات المرجَّحة هي كالتالي: X بنسبة كذا، Y "
            "بنسبة كذا. يُرجى العلم أنّني لست طبيباً ولا يمكنني الجزم بالتشخيص.'\n"
        )
        if include_rag:
            base += (
                "7. RAG_LITERATURE_CONTEXT: مراجع PubMed **مساعدة للشرح فقط** — "
                "لا تُغيّر URGENCY ولا SPECIALTY ولا DISEASE_PREDICTIONS.\n"
            )
        base += (
            "\nأعد ردّك بصيغة JSON فقط، دون أي نصّ خارجه، وبالمفاتيح الأربعة "
            "التالية حصراً (كلّها نصوص عربية غير فارغة):\n"
            '{"summary": "...", "medical_reasoning": "...", '
            '"recommendation": "...", "disclaimer": "..."}'
        )
        return base

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
