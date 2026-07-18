from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request):
    """
    فحص صحّة خدمة المساعد الطبي: حالة نموذج MARBERT (مع بصمة نقطة التحقّق
    المُحمَّلة فعلاً)، جاهزية Whisper، ومزوّد الـ LLM.
    """
    state = request.app.state
    loader = getattr(state, "model_loader", None)
    speech = getattr(state, "speech_service", None)
    conv = getattr(state, "conversation_service", None)

    model_loaded = bool(loader and loader.is_loaded)
    speech_ready = bool(speech and getattr(speech, "is_ready", False))

    provider_name = None
    if conv is not None:
        provider = getattr(conv, "_provider", None)
        provider_name = getattr(provider, "name", None)

    return {
        "status": "ok" if model_loaded else "degraded",
        "message": "Healix Medical Assistant service is running",
        "marbert": {
            "loaded": model_loaded,
            "device": str(loader.device) if loader else None,
            "checkpoint": loader.checkpoint if loader else None,
        },
        "whisper": {"ready": speech_ready},
        "llm_provider": provider_name,
    }
