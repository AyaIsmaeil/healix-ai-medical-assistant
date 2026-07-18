"""
Healix - Interview Prompt Builder
بناء تعليمات الـ LLM لمقابلة أخذ التاريخ المرضي الذكية (بلا تشخيص).

يُنتج:
- ``system_prompt``: دور الطبيب الخبير + إطار OLDCARTS + قواعد صارمة.
- ``turn_prompt``: حالة منظَّمة تشمل **كامل رسائل المريض الخام** (لا تُفقَد أي
  معلومة) + الأعراض المستخرَجة بـ MARBERT (التأريض) + المنفية + الخانات المطروحة
  + الخانات المفقودة المقترحة.

منع فقدان المعلومات: يتلقّى الـ LLM كل ما قاله المريض عبر PATIENT_MESSAGES، فلا
يُعاد السؤال عن تفاصيل ذُكرت في أي دور سابق (كالبداية أو الشدّة).
منع عدم تطابق الخانات: لا نُرسل إطلاقاً خريطة "خانة → قيمة" قد تكون خاطئة؛
الخانات المطروحة (ASKED_SLOTS) لمنع التكرار فقط، والمحتوى الحقيقي في الرسائل.

MARBERT يبقى المصدر الوحيد للأعراض المنظَّمة. عقد المخرجات لم يتغيّر:
{"finished": false, "next_slot": "...", "question": "..."}.
"""

from __future__ import annotations

import json

from app.domain import clinical
from app.domain.conversation import ConversationState

# أسماء الحقول الموسومة (يقرؤها المزوّد الوهمي أيضاً).
LABEL_PATIENT_MESSAGES = "PATIENT_MESSAGES"
LABEL_EXTRACTED = "EXTRACTED_SYMPTOMS"
LABEL_NEGATED = "NEGATED_SYMPTOMS"
LABEL_ASKED = "ASKED_SLOTS"
LABEL_TURN = "TURN_COUNT"
LABEL_SUGGESTED = "SUGGESTED_MISSING_SLOTS"

# عدد الخانات المقترحة التي تُمرَّر للـ LLM كإرشاد.
_SUGGESTION_LIMIT = 6


class InterviewPromptBuilder:
    """يبني تعليمات المقابلة الذكية من حالة المحادثة."""

    version = "interview-v4"

    def system_prompt(self) -> str:
        return (
            "أنت طبيب خبير يُجري أخذ التاريخ المرضي (History Taking) باللغة العربية.\n"
            "مهمتك في هذه المرحلة جمع المعلومات فقط عبر الأسئلة، ولست مطالباً بأي تشخيص.\n\n"
            "في كل دور تتلقّى حالة منظَّمة تحتوي:\n"
            "- PATIENT_MESSAGES: كل ما قاله المريض بترتيب الزمن (آخرها الأحدث). هذه "
            "هي المعلومات الكاملة الموثوقة عمّا ذكره المريض.\n"
            "- EXTRACTED_SYMPTOMS و NEGATED_SYMPTOMS: الأعراض التي استخرجها نموذج "
            "طبي (MARBERT) — هذه هي التأريض المعتمد للأعراض.\n"
            "- ASKED_SLOTS: الخانات التي سبق طرح سؤال عنها (لمنع تكرار السؤال).\n"
            "- TURN_COUNT و SUGGESTED_MISSING_SLOTS.\n\n"
            "اعتمد الأعراض من EXTRACTED_SYMPTOMS/NEGATED_SYMPTOMS، واقرأ "
            "PATIENT_MESSAGES بعناية لمعرفة كل ما أخبرك به المريض فعلاً.\n\n"
            "فكّر كطبيب متمرّس: لكل عرَض مهم اجمع onset، duration، severity، progression، "
            "location، quality، associated symptoms، aggravating و relieving factors. "
            "واسأل عند اللزوم فقط عن العمر، الجنس، الحمل، الأمراض المزمنة، الأدوية، الحساسية، "
            "التدخين والعمليات — إن كانت ذات صلة طبية. واجعل الأسئلة ديناميكية حسب العرَض.\n\n"
            "القواعد الإلزامية:\n"
            "1. اطرح سؤالاً واحداً فقط في كل ردّ.\n"
            "2. يجب أن يكون السؤال بعربية طبيعية ومهذّبة.\n"
            "3. اختر السؤال الأعلى قيمة تشخيصية الذي يقلّل الغموض أكثر من غيره.\n"
            "4. لا تكرّر أي سؤال أو خانة وردت في ASKED_SLOTS.\n"
            "5. لا تسأل إطلاقاً عن أي معلومة ذُكرت في أيٍّ من PATIENT_MESSAGES "
            "(مثل بداية الأعراض أو شدّتها أو توقيتها إن سبق ذكرها في أي دور).\n"
            "6. لا تقدّم أي تشخيص أو احتمالات أمراض أو تخصّص أو درجة خطورة أو توصية.\n"
            "7. أعد finished=true فقط عندما يصبح التاريخ المرضي كافياً (لا معلومات مهمة ناقصة).\n\n"
            "استعن بـ SUGGESTED_MISSING_SLOTS كإرشاد، لكن تجاوز أي خانة تكون إجابتها "
            "مذكورة أصلاً في PATIENT_MESSAGES. اسمِّ الخانة في next_slot بنفس صيغة الأهداف.\n\n"
            "أعد ردّك بصيغة JSON فقط، دون أي نصّ خارج JSON، وبالشكل التالي حصراً:\n"
            '{"finished": false, "next_slot": "<الهدف>", "question": "<السؤال بالعربية>"}\n'
            "أو عند الاكتفاء:\n"
            '{"finished": true}'
        )

    def turn_prompt(self, state: ConversationState) -> str:
        extracted = [s.text for s in state.symptoms if not s.negated]
        negated = [s.text for s in state.symptoms if s.negated]
        # التغطية للتكرار: خانة تُعدّ مغطّاة بمجرّد طرح سؤال عنها.
        suggested = clinical.missing_targets(
            state.symptoms,
            {},
            state.asked_slots,
            limit=_SUGGESTION_LIMIT,
        )

        def line(label: str, value) -> str:
            return f"{label}: {json.dumps(value, ensure_ascii=False)}"

        state_block = "\n".join([
            line(LABEL_PATIENT_MESSAGES, state.raw_messages),
            line(LABEL_EXTRACTED, extracted),
            line(LABEL_NEGATED, negated),
            line(LABEL_ASKED, state.asked_slots),
            line(LABEL_TURN, state.turn_count),
            line(LABEL_SUGGESTED, suggested),
        ])

        return (
            "الحالة المنظَّمة الحالية للمقابلة:\n"
            f"{state_block}\n\n"
            "اعتماداً على هذه الحالة: حدّد الخانة الأعلى قيمة تشخيصية التي لم "
            "تُغطَّ بعد ولم يذكرها المريض في أيٍّ من PATIENT_MESSAGES، واطرح سؤالاً "
            "عربياً واحداً عنها. إن اكتمل جمع التاريخ المرضي فأعد finished=true. "
            "أعد JSON فقط."
        )
