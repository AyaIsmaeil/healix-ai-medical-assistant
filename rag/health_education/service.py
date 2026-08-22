"""Orchestrates safety_gate + retriever + LLM summary into a
HealthQuestionResponse. The single entry point api/main.py's
POST /health-questions route calls.

Never touches graph.py/state.py or any triage node. If everything here
fails (LLM unavailable, retriever error), that failure stays contained
to this function/route — POST /chat shares no state with it and cannot
be affected (see tests/unit/test_health_qa.py's
test_module_failure_does_not_affect_chat-style coverage).
"""

from __future__ import annotations

from llm_client import call_llm
from nodes._shared import support_line_text
from prompts.base import build_prompt
from rag.health_education.retriever import MIN_SCORE, HealthEducationRetriever, get_retriever
from schemas.health_education import HealthEducationAnswer

from api.health_qa_contracts import HealthQuestionResponse, SourceReference
from rag.health_education.safety_gate import classify

DISCLAIMER = (
    "هاد المحتوى معلومات تثقيفية عامة فقط، مو تشخيص طبي ومو وصفة علاجية. "
    "لأي قرار طبي شخصي، راجع طبيبك."
)

_TRIAGE_REDIRECT_MESSAGE = (
    "هاد الموضوع خاص بحالتك الشخصية، وما بينجاوب عليه هون. لتقييم دقيق "
    "لعرضك، رجاءً استخدم محادثة التقييم الطبي (Healix) يلي مبنية لهيك حالات."
)

_MEDICATION_SAFETY_MESSAGE = (
    "ما بقدر أعطيك اسم دواء أو جرعة أو توصية علاجية من هون — هاد قرار لازم "
    "ياخده طبيب أو صيدلاني بعد معاينة حالتك. إذا عندك عرض حالي، رجاءً "
    "استخدم محادثة التقييم الطبي (Healix)."
)

_INSUFFICIENT_INFO_MESSAGE = (
    "ما عندي معلومات كافية موثوقة للإجابة عن هالسؤال بدقة حاليًا. ممكن "
    "تجرب تعيد صياغة السؤال، أو تراجع طبيبك مباشرة."
)


def _emergency_redirect_message() -> str:
    return (
        "الوصف يلي كتبتو ممكن يشير لحالة إسعافية. ما بقدر جاوبك هون — "
        "توجه فورًا لأقرب قسم طوارئ، أو تواصل معنا عبر محادثة التقييم "
        "الطبي (Healix) لتقييم فوري:\n" + support_line_text()
    )


def _format_excerpts(hits: list) -> str:
    """Grounding text handed to the LLM prompt only — never returned to
    the caller (api.health_qa_contracts.HealthQuestionResponse has no
    field for it). See tests/unit/test_health_qa.py's no-raw-leak tests."""
    lines = []
    for hit in hits:
        lines.append(f"- الفئة: {hit.category}\n  السؤال: {hit.question}\n  الإجابة: {hit.answer}")
    return "\n".join(lines)


def _unique_sources(hits: list, limit: int = 3) -> list[SourceReference]:
    seen: list[str] = []
    for hit in hits:
        if hit.category not in seen:
            seen.append(hit.category)
        if len(seen) >= limit:
            break
    return [SourceReference(category=category) for category in seen]


def answer_health_question(
    question: str,
    *,
    thread_id: str | None = None,
    retriever: HealthEducationRetriever | None = None,
) -> HealthQuestionResponse:
    """One health-education turn. `retriever` is injectable for tests;
    production omits it and get_retriever() lazy-loads the real corpus."""
    category = classify(question, thread_id=thread_id)

    if category == "emergency_redirect":
        return HealthQuestionResponse(
            answer=_emergency_redirect_message(),
            category="emergency_redirect",
            sources=[],
            grounded=False,
            disclaimer=DISCLAIMER,
            retrieval_status="insufficient",
        )
    if category == "triage_redirect":
        return HealthQuestionResponse(
            answer=_TRIAGE_REDIRECT_MESSAGE,
            category="triage_redirect",
            sources=[],
            grounded=False,
            disclaimer=DISCLAIMER,
            retrieval_status="insufficient",
        )
    if category == "medication_safety":
        return HealthQuestionResponse(
            answer=_MEDICATION_SAFETY_MESSAGE,
            category="medication_safety",
            sources=[],
            grounded=False,
            disclaimer=DISCLAIMER,
            retrieval_status="insufficient",
        )

    active_retriever = retriever if retriever is not None else get_retriever()
    hits = [hit for hit in active_retriever.search(question, top_k=5) if hit.score >= MIN_SCORE]

    if not hits:
        return HealthQuestionResponse(
            answer=_INSUFFICIENT_INFO_MESSAGE,
            category="out_of_scope",
            sources=[],
            grounded=False,
            disclaimer=DISCLAIMER,
            retrieval_status="insufficient",
        )

    prompt = build_prompt("health_qa_answer", question=question, excerpts=_format_excerpts(hits))
    llm_result = call_llm(
        prompt,
        schema=HealthEducationAnswer,
        tier="fast",
        prompt_name="health_qa_answer",
        thread_id=thread_id,
    )
    assert isinstance(llm_result, HealthEducationAnswer)

    return HealthQuestionResponse(
        answer=llm_result.answer,
        category="educational",
        sources=_unique_sources(hits),
        grounded=True,
        disclaimer=DISCLAIMER,
        retrieval_status="sufficient",
    )
