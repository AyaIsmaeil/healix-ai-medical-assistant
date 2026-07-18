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
