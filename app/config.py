import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DISEASES_PATH = os.path.join(DATABASE_DIR, "diseases.json")
SYMPTOMS_PATH = os.path.join(DATABASE_DIR, "symptoms.json")

# Model and service settings
MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

# MARBERT symptom-extraction model (Phase 1).
# Threshold and max_length are read from the model's metadata.json; the value
# below only overrides them if explicitly set via environment variable.
SYMPTOM_MODEL_DIR = os.getenv(
    "SYMPTOM_MODEL_DIR",
    os.path.normpath(os.path.join(BASE_DIR, "..", "ml_models", "healix_marbert_v2")),
)

# API settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Whisper speech-to-text settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ar")
MAX_AUDIO_FILE_SIZE_MB = int(os.getenv("MAX_AUDIO_FILE_SIZE_MB", "25"))
AUDIO_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("AUDIO_DOWNLOAD_TIMEOUT_SECONDS", "30"))

ALLOWED_AUDIO_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.getenv(
        "ALLOWED_AUDIO_EXTENSIONS",
        ".wav,.mp3,.m4a,.ogg,.flac,.webm,.mp4,.mpeg,.mpga",
    ).split(",")
    if ext.strip()
}
