"""
Healix - Domain Exceptions
استثناءات مخصّصة لطبقة الاستدلال (Inference).

فصل الأخطاء في أنواع واضحة يسهّل معالجتها في طبقة الـ API
ويجعل المنطق قابلاً للاختبار دون الاعتماد على استثناءات عامة.
"""


class HealixError(Exception):
    """الاستثناء الأساسي لكل أخطاء خدمة Healix."""


class ModelLoadError(HealixError):
    """يُرفع عند فشل تحميل النموذج أو الـ Tokenizer."""


class ModelNotLoadedError(HealixError):
    """يُرفع عند طلب الاستدلال قبل تحميل النموذج."""


class InferenceError(HealixError):
    """يُرفع عند فشل تنفيذ عملية الاستدلال على النص."""


class ConversationError(HealixError):
    """يُرفع عند فشل في محرك المحادثة (History Taking)."""


class LLMProviderError(HealixError):
    """يُرفع عند فشل استدعاء مزوّد الـ LLM."""

    # هل الخطأ عابر ويصلح لإعادة المحاولة؟ (يُعلَّم صراحةً عند الرفع)
    retryable: bool = False


class InterviewParsingError(HealixError):
    """يُرفع عند فشل تحويل مخرجات الـ LLM إلى JSON صالح للمقابلة."""


# ----------------------------------------------------------------------
# أخطاء الصوت / التفريغ النصّي (Whisper Speech-to-Text)
# ----------------------------------------------------------------------
class SpeechError(HealixError):
    """الأساس لأخطاء معالجة الصوت."""


class InvalidAudioInputError(SpeechError):
    """مدخل صوتي غير صالح (مسار فارغ، أو لا مسار ولا رابط...)."""


class AudioNotFoundError(SpeechError):
    """ملف الصوت غير موجود أو غير قابل للقراءة."""


class UnsupportedAudioFormatError(SpeechError):
    """صيغة صوتية غير مدعومة."""


class AudioTooLargeError(SpeechError):
    """حجم ملف الصوت يتجاوز الحد المسموح."""


class AudioDownloadError(SpeechError):
    """فشل تنزيل الصوت من رابط."""


class TranscriptionError(SpeechError):
    """فشل التفريغ النصّي بواسطة Whisper."""


# ----------------------------------------------------------------------
# أخطاء محرك التقييم (Assessment Engine — Phase 3.1)
# ----------------------------------------------------------------------
class AssessmentError(HealixError):
    """الأساس لأخطاء محرك التقييم."""


class FeatureExtractionError(AssessmentError):
    """يُرفع عند فشل استخراج الميزات (قاعدي أو عبر LLM) بشكل غير قابل للتعافي."""


class FeatureValidationError(HealixError):
    """يُرفع عند فشل طبقة التحقّق من الميزات (Phase 3.2).

    يغطّي حالتين مترابطتين: (1) قاموس قواعد التحقّق مشوّه أو غير موجود عند
    الإقلاع (DictionaryLoader)، و(2) غياب معلومة سريرية جوهرية وقت التحقّق
    الفعلي (FeatureValidator) — أي حقل اختياري غير صالح يُعالَج بفشل ناعم
    (تصحيح/null) بلا رفع استثناء."""


class AssessmentExplanationError(AssessmentError):
    """يُرفع عند فشل تحليل مخرجات الـLLM لتفسير التقييم (Phase 3.8) إلى JSON
    صالح بالعقد المطلوب (المفاتيح الأربعة نصوصاً غير فارغة).

    يرث من ``AssessmentError`` (ومن ثمّ ``HealixError``) ليُشغِّل آلية إعادة
    المحاولة + التلقين المدمجة بمزوّد الـLLM (التي تلتقط ``HealixError``).
    ملاحظة: خدمة ``AssessmentExplainer`` تلتقط هذا الخطأ وتتدهور بلطف إلى
    تفسير حتمي بديل — فلا يتسرّب هذا الاستثناء للراوت ولا يُسقط التقييم
    المحسوب سلفاً (نفس فلسفة التدهور اللطيف بـ``LLMFeatureExtractor``)."""
