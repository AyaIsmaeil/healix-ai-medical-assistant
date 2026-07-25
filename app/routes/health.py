from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request):
    """
    فحص صحّة خدمة المساعد الطبي: جاهزية مزوّد الـLLM (العقل الأساسي للمساعد
    بعد الانتقال لمعمارية LLM-first)، وجاهزية Whisper.

    ``status`` مشتقّ من جاهزية الـLLM لأنّه صار المكوّن الحرج: بدونه لا مقابلة
    ولا استخراج. Whisper يبقى اختيارياً (يعطّل الصوت فقط، لا النصّ) فلا يُخفّض
    الحالة العامة.
    """
    state = request.app.state
    speech = getattr(state, "speech_service", None)
    conv = getattr(state, "conversation_service", None)

    speech_ready = bool(speech and getattr(speech, "is_ready", False))

    provider = getattr(conv, "_provider", None) if conv is not None else None
    provider_name = getattr(provider, "name", None)

    # المزوّد قد يوفّر فحصاً منظَّماً (mock/openrouter). غيابه لا يُعدّ فشلاً:
    # نعود عندها لجاهزية المحرك نفسه كمؤشّر أدنى.
    llm_health = None
    health_fn = getattr(provider, "health", None)
    if callable(health_fn):
        try:
            llm_health = health_fn()
        except Exception as exc:  # noqa: BLE001 - الفحص لا يُسقط نقطة الصحّة
            llm_health = {"provider": provider_name, "ok": False, "error": str(exc)}

    if llm_health is not None:
        llm_ready = bool(llm_health.get("ok"))
    else:
        llm_ready = conv is not None

    return {
        "status": "ok" if llm_ready else "degraded",
        "message": "Healix Medical Assistant service is running",
        "llm": {
            "ready": llm_ready,
            "provider": provider_name,
            "details": llm_health,
        },
        "whisper": {"ready": speech_ready},
        "llm_provider": provider_name,
    }
