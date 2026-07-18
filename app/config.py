import os
from dotenv import load_dotenv
from pathlib import Path

# تحميل متغيرات البيئة
load_dotenv()

# مسارات المشروع
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

class Config:
    # ------------------------------------------------------------------
    # النموذج (MARBERT Token-Classification / NER لاستخراج الأعراض)
    # ------------------------------------------------------------------
    # مصدر النموذج: مجلد محلي إن وُجد، وإلا مستودع Hugging Face.
    LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "marbert-ner"))
    HF_MODEL_ID = os.getenv("HF_MODEL_ID", "ayaismael/marbert-symptom-ner")

    MODEL_NAME = os.getenv("MODEL_NAME", "UBC-NLP/MARBERTv2")
    MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 256))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

    def model_source(self) -> str:
        """
        تحديد مصدر النموذج المُدرّب.
        يُفضّل المجلد المحلي (إن كان يحتوي على ملفات النموذج) وإلا يُستخدم
        مُعرّف مستودع Hugging Face حتى تعمل الخدمة دون الحاجة لرفع الأوزان.
        """
        local = Path(self.LOCAL_MODEL_PATH)
        if local.is_dir() and (local / "config.json").exists():
            return str(local)
        return self.HF_MODEL_ID

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"
    
 
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
    

    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    
   
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    

    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    TOP_K_PREDICTIONS = int(os.getenv("TOP_K_PREDICTIONS", 3))

    # ------------------------------------------------------------------
    # محرك المحادثة (Conversation Engine) + مزوّد الـ LLM
    # ------------------------------------------------------------------
    # تفعيل تحميل Whisper ضمن الخدمة الموحّدة (يمكن تعطيله لتسريع بدء التطوير).
    ENABLE_WHISPER = os.getenv("ENABLE_WHISPER", "true").lower() == "true"

    # المزوّد: "mock" أو "qwen_openrouter" — يُبدَّل من الإعدادات فقط دون تغيير أي منطق.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    # الحد الأقصى لعدد الأسئلة قبل إنهاء المقابلة (حماية من الحلقات اللانهائية).
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 20))

    # ------------------------------------------------------------------
    # OpenRouter (Qwen3) — كل القيم من متغيّرات البيئة، لا شيء ثابت في الكود
    # ------------------------------------------------------------------
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-14b")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # مهلة طلب OpenRouter بالثواني (OPENROUTER_TIMEOUT، مع LLM_TIMEOUT كتوافق خلفي).
    OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", os.getenv("LLM_TIMEOUT", 60)))

    # وضع المخرجات المنظَّمة: "schema" (json_schema صارم) | "object" (json_object)
    # | "off" (الاعتماد على التلقين + إعادة المحاولة فقط).
    # عند رفض النموذج للوضع الأعلى يُخفَّض تلقائياً درجة واحدة.
    OPENROUTER_JSON_MODE = os.getenv("OPENROUTER_JSON_MODE", "schema").strip().lower()

    # عند تعذّر الوصول لـ OpenRouter: الرجوع تلقائياً للمزوّد الوهمي (مع تحذير).
    LLM_FALLBACK_TO_MOCK = os.getenv("LLM_FALLBACK_TO_MOCK", "false").lower() == "true"

    # توليد حتمي منخفض الحرارة (مقابلة طبية فقط، JSON فقط).
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))
    # عدد المحاولات الكلي عند JSON غير صالح أو خطأ عابر (محاولة + إعادة).
    LLM_JSON_ATTEMPTS = int(os.getenv("LLM_JSON_ATTEMPTS", 3))

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