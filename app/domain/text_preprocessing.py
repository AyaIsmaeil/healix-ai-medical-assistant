"""
Healix - Arabic Text Preprocessing (تطبيع النصّ العربي المُدخَل)

مرحلة معالجة أوّليّة للنصّ العربي قبل المطابقة المعجمية الحتمية (الأعلام
الحمراء، الأولوية السريرية، ربط الأدلة). الغرض جعل الصيغ الشكلية المختلفة
لنفس الكلمة متطابقةً ("ألم صَدر" = "الم صدر") دون تغيير المعنى.

**اعتماد مكتبي موثَّق:** يعتمد التطبيع اللغوي الأساسي على **CAMeL Tools**
(Obeid et al., LREC 2020) — المرجع المعياري لمعالجة اللغة العربية الحاسوبية —
تحديداً وحدات ``camel_tools.utils.normalize`` و``camel_tools.utils.dediac``
(أدوات خفيفة قائمة على regex، لا تُحمّل أي نموذج صرفي ثقيل). أُضيفت حولها
خطوات حتمية مكمّلة (تحويل الأرقام العربية-الهندية، توحيد الهمزات المتبقية،
تنظيف الترقيم والمسافات) لتغطية ما لا تشمله دوال CAMeL. الأحرف اللاتينية
(مثل "COVID") **تبقى نصّاً لاتينياً ولا تُشوَّه إلى عربية**، وإن طُبِّع شكلها
للمطابقة (تصغير الأحرف، والشرطة تُعامَل كترقيم فيصير "COVID-19" → "covid 19").

قرار تصميم صريح (مبنيّ على اختبار حيّ): **لا يُطبَّق هذا التطبيع قبل النموذج
اللغوي (LLM)** — فقد ثبت تجريبياً أنّ Qwen3 يفهم اللهجات والأخطاء الإملائية
الخام أفضل من أي تنظيف مسبق، وتنظيفُ النصّ قبله قد يحذف إشارةً مفيدة. التطبيع
هنا مخصَّص **للطبقات الحتمية فقط**.

قيدٌ معلَن: التطبيع عمليةٌ **شكلية** لا تصحّح الإملاء ("قووي" تبقى "قووي")
ولا تترجم اللهجة إلى الفصحى ("عم" تبقى "عم") — تلك مهامٌّ مختلفة (تصحيح
إملائي/ترجمة لهجية) خارج نطاق التطبيع، ويتكفّل بها النموذج اللغوي.
"""

from __future__ import annotations

import re
import unicodedata

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

# همزات وأرقام لا تغطّيها دوال CAMeL أعلاه — تُكمَّل حتمياً هنا:
# ``normalize_alef_ar`` يوحّد (أ إ آ ٱ)→ا، و``normalize_alef_maksura_ar``
# (ى)→ي، و``normalize_teh_marbuta_ar`` (ة)→ه؛ لكنّ الهمزة على الياء/الواو
# والأرقام العربية-الهندية خارج نطاقها.
_RESIDUAL_MAP = str.maketrans({
    "ئ": "ي",
    "ؤ": "و",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})

# التشكيل والتطويل — يحذفهما ``dediac_ar``؛ نُبقي هذا الحارس احتياطاً
# (idempotent) لأيّ محرف تشكيل نادر خارج تغطية CAMeL.
_RESIDUAL_DIACRITICS = re.compile(r"[ً-ْٰـ]")

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """يُطبِّع نصّاً عربياً/إنجليزياً للمطابقة المعجمية الحتمية.

    يوحّد الألف والياء والتاء المربوطة والهمزات (عبر CAMeL Tools + إكمال
    حتمي)، يحذف التشكيل والتطويل، يحوّل الأرقام العربية-الهندية، يزيل الترقيم،
    ويوحّد المسافات. لا يغيّر المعنى — عملياتٌ شكلية بحتة.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # CAMeL: التطبيع اللغوي الأساسي (الأساس الموثَّق مكتبياً).
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    # إكمال حتمي لما لا تشمله دوال CAMeL:
    text = _RESIDUAL_DIACRITICS.sub("", text)
    text = text.translate(_RESIDUAL_MAP)
    text = _NON_WORD.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().lower()
