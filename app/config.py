import os
from dotenv import load_dotenv
from pathlib import Path

# تحميل متغيرات البيئة
load_dotenv()

# مسارات المشروع
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

class Config:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

    # ------------------------------------------------------------------
    # وكيل المقابلة السريرية (Clinical Interview) + مزوّد الـ LLM
    # ------------------------------------------------------------------
    # تفعيل تحميل Whisper ضمن الخدمة الموحّدة (يمكن تعطيله لتسريع بدء التطوير).
    ENABLE_WHISPER = os.getenv("ENABLE_WHISPER", "true").lower() == "true"

    # المزوّد: "mock" أو "qwen_openrouter" — يُبدَّل من الإعدادات فقط دون تغيير أي منطق.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    # الحد الأقصى لعدد الأسئلة قبل إنهاء المقابلة (حماية من الحلقات اللانهائية).
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 20))

    # انتهاء صلاحية جلسة المقابلة (بالثواني) وحدّ الجلسات النشطة.
    # الجلسات تحمل بيانات صحّية شخصية، فبقاؤها بلا انتهاء تسرّب موارد
    # ومشكلة خصوصية معاً. الصلاحية تُحتسب من آخر نشاط لا من الإنشاء.
    SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))
    MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", 1000))

    # ------------------------------------------------------------------
    # OpenRouter (Qwen3) — كل القيم من متغيّرات البيئة، لا شيء ثابت في الكود
    # ------------------------------------------------------------------
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    # Qwen3-32B: العقل الأساسي للمساعد بعد الانتقال لمعمارية LLM-first
    # (استخراج منظَّم + أسئلة متابعة في استدعاء واحد).
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-32b")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # مهلة طلب OpenRouter بالثواني (OPENROUTER_TIMEOUT، مع LLM_TIMEOUT كتوافق خلفي).
    OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", os.getenv("LLM_TIMEOUT", 60)))

    # وضع المخرجات المنظَّمة: "schema" (json_schema صارم) | "object" (json_object)
    # | "off" (الاعتماد على التلقين + إعادة المحاولة فقط).
    # عند رفض النموذج للوضع الأعلى يُخفَّض تلقائياً درجة واحدة.
    OPENROUTER_JSON_MODE = os.getenv("OPENROUTER_JSON_MODE", "schema").strip().lower()

    # وضع تفكير النموذج (Qwen3 thinking). الافتراضي false = مُعطَّل → ردود سريعة
    # جداً (بلا رموز تفكير مطوّلة). فعّله فقط إن أردت جودة أعلى على حساب السرعة.
    OPENROUTER_REASONING = os.getenv("OPENROUTER_REASONING", "false").lower() == "true"

    # عند تعذّر الوصول لـ OpenRouter: الرجوع تلقائياً للمزوّد الوهمي (مع تحذير).
    LLM_FALLBACK_TO_MOCK = os.getenv("LLM_FALLBACK_TO_MOCK", "false").lower() == "true"

    # توليد حتمي منخفض الحرارة (مقابلة طبية فقط، JSON فقط).
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    # رُفع من 512 مع العقد الموحّد: صار الردّ الواحد يحمل السجل الطبي المنظَّم
    # كاملاً (أعراض + قوائم أدوية/حساسية/مزمنة/عائلي) بالعربية **مع** السؤال،
    # فسقف 512 كان يُخاطر بقطع الـJSON في منتصفه.
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))
    # عدد المحاولات الكلي عند JSON غير صالح أو خطأ عابر (محاولة + إعادة).
    LLM_JSON_ATTEMPTS = int(os.getenv("LLM_JSON_ATTEMPTS", 3))

    # ------------------------------------------------------------------
    # مُتنبِّئ المرض — مسار ML موازٍ (Phase 3.4)
    # ------------------------------------------------------------------
    # عند التفعيل: MLDiseasePredictor (models/) يُحقَن بدل RuleBasedDiseasePredictor
    # القاعدي — بلا حذف أو تعديل الأخير، فقط تفرّع بـmain.py.
    USE_ML_PREDICTOR = os.getenv("USE_ML_PREDICTOR", "false").lower() == "true"

class DevelopmentConfig(Config):
    """إعدادات بيئة التطوير"""
    RELOAD = True
    LOG_LEVEL = "debug"

class ProductionConfig(Config):
    """إعدادات بيئة الإنتاج"""
    RELOAD = False
    LOG_LEVEL = "warning"

# اختيار الإعدادات حسب البيئة
ENV = os.getenv("ENV", "development")
if ENV == "production":
    config = ProductionConfig()
else:
    config = DevelopmentConfig()

# ----------------------------------------------------------------------
# إعدادات Whisper (تفريغ الصوت) — ثوابت على مستوى الوحدة يستهلكها
# خدمة SpeechToTextService مباشرةً.
# ----------------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ar")
MAX_AUDIO_FILE_SIZE_MB = int(os.getenv("MAX_AUDIO_FILE_SIZE_MB", 25))
AUDIO_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("AUDIO_DOWNLOAD_TIMEOUT_SECONDS", 30))
ALLOWED_AUDIO_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.getenv(
        "ALLOWED_AUDIO_EXTENSIONS", ".wav,.mp3,.m4a,.ogg,.flac,.webm,.mp4,.mpga"
    ).split(",")
    if ext.strip()
}

# تصدير الإعدادات
__all__ = [
    "config",
    "Config",
    "WHISPER_MODEL",
    "WHISPER_LANGUAGE",
    "MAX_AUDIO_FILE_SIZE_MB",
    "AUDIO_DOWNLOAD_TIMEOUT_SECONDS",
    "ALLOWED_AUDIO_EXTENSIONS",
]